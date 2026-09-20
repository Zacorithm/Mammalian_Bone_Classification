"""Entry point: training (config split or --train_mode cv), testing, and registered analyses.

    python main.py --config <cfg> --exp_name <name>                      # train on the config split
    python main.py --config <cfg> --train_mode cv --cv_folds 5 ...       # K-fold CV
    python main.py --config <cfg> --pretrain_config <cfg> --train_mode cv # SSL: pretrain+finetune per fold
    python main.py --config <cfg> --dvae_config <cfg> ...                 # + PointBERT dVAE stage
    python main.py --config <cfg> --test --ckpts <ckpt-best.pth>
    python main.py --config <cfg> --analysis umap_tsne --ckpts <ckpt-best.pth>
Add --resume to continue a crashed run (each fold/stage resumes from its own checkpoint).
"""

import copy
import json
import os
import shutil
import time

import torch
from tensorboardX import SummaryWriter

from tools import pretrain_run_net as pretrain
from tools import pretrain_run_net_BERT as pretrain_BERT
from tools import pretrain_dvae_BERT as pretrain_dvae_BERT
from tools import finetune_run_net as finetune
from tools import finetune_run_net_GPT as finetune_GPT
from tools import test_run_net as test_net
from tools import test_run_net_GPT as test_net_GPT
from tools import generate_visualizations, visualize_bone_taxonomy
from datasets.BoneDataset import compute_cv_fold_split
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *


# Analyses registry: register a (args, config) -> None callable here and add its
# name to ANALYSIS_CHOICES in utils/parser.py.
ANALYSES = {
    'umap_tsne': lambda args, config: generate_visualizations(args, config),
    'taxonomy': lambda args, config: visualize_bone_taxonomy(
        args, config, bone_name=args.vis_bone, split=args.vis_split),
}

# per-fold metrics aggregated by cross-validation (ComprehensiveMetrics fields)
CV_METRIC_FIELDS = ['overall_acc', 'class_acc', 'recall', 'precision', 'f1_score']
# per-fold inference timing aggregated alongside them (ComprehensiveMetrics.timing keys)
CV_TIMING_FIELDS = ['mean_inference_time_ms_per_sample', 'mean_inference_time_s_per_batch']
# per-class fields averaged across folds' best epochs (see calculate_comprehensive_metrics)
CLASS_METRIC_FIELDS = ['accuracy', 'precision', 'recall', 'f1_score', 'support']


def print_run_summary(args, config, logger):
    """Short, console-visible summary of what's about to run — the full
    args/config dump still goes to the log file (see log_args_to_file /
    log_config_to_file), just not to the terminal."""
    print_log(f'[main] config: {args.config}', logger=logger)
    print_log(f'[main] exp_name: {args.exp_name}  ->  {args.experiment_path}', logger=logger)
    print_log(f'[main] model: {config.model.NAME}', logger=logger)
    opt = config.optimizer
    sch = config.scheduler
    print_log(f'[main] optimizer: {opt.type} (lr={opt.kwargs.lr}, '
              f'weight_decay={opt.kwargs.get("weight_decay", 0)})  '
              f'scheduler: {sch.type}', logger=logger)
    print_log(f'[main] total_bs={config.total_bs}  npoints={config.npoints}  '
              f'max_epoch={config.max_epoch}', logger=logger)
    mode_str = args.train_mode
    if args.train_mode == 'cv':
        mode_str += f' (cv_folds={args.cv_folds}, seed={args.seed})'
    stage = 'test' if args.test else (
        'finetune' if args.finetune_model else ('scratch' if args.scratch_model else 'auto'))
    print_log(f'[main] train_mode: {mode_str}  stage: {stage}  resume: {args.resume}',
              logger=logger)


def _is_gpt(config):
    """PointGPT models return (logits, aux_loss) and need the dedicated runner."""
    return config.model.NAME.startswith('PointGPT')


def _is_pretrain(config):
    return config.model.NAME.endswith('_Pretrain')


def _assign_batch_sizes(args, config):
    """Fill the per-loader batch sizes from config.total_bs."""
    if args.distributed:
        assert config.total_bs % args.world_size == 0
        per_gpu = config.total_bs // args.world_size
    else:
        per_gpu = config.total_bs
    config.dataset.train.others.bs = per_gpu
    if config.dataset.get('extra_train'):
        config.dataset.extra_train.others.bs = per_gpu * 2
    config.dataset.val.others.bs = per_gpu * 2
    if config.dataset.get('test'):
        config.dataset.test.others.bs = per_gpu


def _apply_fewshot_args(args, config):
    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold


def _build_writers(args):
    """Tensorboard writers (rank 0 only)."""
    if args.local_rank == 0:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    else:
        train_writer = None
        val_writer = None
    return train_writer, val_writer


def run_training(args, config, logger):
    """Train one model on one config. Returns the best validation metrics
    (ComprehensiveMetrics) for classification runs, None for pretraining."""
    train_writer, val_writer = _build_writers(args)

    if _is_pretrain(config):
        if config.model.NAME == 'PointBERT_Pretrain':
            pretrain_BERT(args, config, train_writer, val_writer)
        elif config.model.NAME == 'PointBERT_DVAE_Pretrain':
            pretrain_dvae_BERT(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)
        return None

    # default to scratch when no flag was given
    if not (args.finetune_model or args.scratch_model):
        args.scratch_model = True
        print_log('[main] no --finetune_model/--scratch_model given for a '
                  '_Classification model: training from scratch', logger=logger)

    if _is_gpt(config):
        return finetune_GPT(args, config, train_writer, val_writer)
    return finetune(args, config, train_writer, val_writer)


def _load_best_epoch_class_metrics(fold_experiment_path, best_epoch):
    """Per-class metrics for one fold's best validation epoch, read from that
    fold's epoch_metrics.json. Returns None if the file/epoch is unavailable."""
    if best_epoch is None:
        return None
    metrics_file = os.path.join(fold_experiment_path, 'epoch_metrics.json')
    if not os.path.exists(metrics_file):
        return None
    with open(metrics_file) as f:
        epoch_metrics = json.load(f)
    return epoch_metrics.get('valid', {}).get(str(best_epoch))


def _aggregate_class_metrics(per_fold_class_metrics):
    """Average each class's metrics across the folds that had usable data.
    per_fold_class_metrics: list of {class_name: {metric: value}}, one dict
    per fold (already filtered to non-None)."""
    classes = sorted({cls for fold in per_fold_class_metrics for cls in fold})
    summary = {}
    for cls in classes:
        per_fold_values = [fold[cls] for fold in per_fold_class_metrics if cls in fold]
        agg = {}
        for field in CLASS_METRIC_FIELDS:
            values = [v[field] for v in per_fold_values if field in v]
            agg[field] = sum(values) / len(values) if values else float('nan')
        agg['folds_used'] = len(per_fold_values)
        summary[cls] = agg
    return summary


def _make_stage_args(args, exp_name, experiment_path, tfboard_path):
    """Shallow-copy args for one stage of one fold, with its own dirs."""
    stage_args = copy.copy(args)
    stage_args.exp_name = exp_name
    stage_args.experiment_path = experiment_path
    stage_args.tfboard_path = tfboard_path
    parser.create_experiment_dir(stage_args)
    return stage_args


def _stage_has_checkpoint(stage_experiment_path):
    return os.path.exists(os.path.join(stage_experiment_path, 'ckpt-last.pth'))


def _set_stage_resume(stage_args, global_resume, logger, stage_label):
    """Resume this stage from its own ckpt-last.pth iff --resume was passed
    AND that checkpoint already exists. A stage/fold with no prior checkpoint
    always starts fresh regardless of --resume — so within one --resume
    invocation, already-finished folds/stages fall through almost instantly
    (the runner sees start_epoch > max_epoch), a partially-trained one
    continues from its last saved epoch, and a fold that never started
    trains normally from scratch."""
    stage_args.resume = bool(global_resume) and _stage_has_checkpoint(stage_args.experiment_path)
    if stage_args.resume:
        print_log(f'[main] {stage_label}: resuming from existing checkpoint '
                  f'in {stage_args.experiment_path}', logger=logger)
    return stage_args.resume


def _inject_cv_split(dataset_cfg, fold, cv_folds, cv_seed, metadata_path=None):
    """Stamp the fold parameters into every split of a config's dataset
    section. metadata_path (SSL pipeline only) additionally overrides the
    stage's metadata CSV so every stage of a fold splits the exact same
    pool of samples."""
    for split in ('train', 'val', 'test'):
        section = dataset_cfg.get(split)
        if section is None:
            continue
        section.others.cv_fold = fold
        section.others.cv_folds = cv_folds
        section.others.cv_seed = cv_seed
        if metadata_path is not None:
            name = section._base_.NAME
            key = 'METADATA_FILE_PATH' if name == 'BoneDataset' else 'METADATA_PATH'
            section.others[key] = metadata_path


def _load_stage_config(stage_args, config_path):
    """Load and prepare a pretraining-stage config (batch sizes + a copy in the stage dir).
    On resume with an existing checkpoint, reload the saved config.yaml so yaml edits can't leak in."""
    saved_path = os.path.join(stage_args.experiment_path, 'config.yaml')
    if stage_args.resume and os.path.exists(saved_path):
        stage_config = cfg_from_yaml_file(saved_path)
    else:
        stage_config = cfg_from_yaml_file(config_path)
        shutil.copy(config_path, saved_path)
    _assign_batch_sizes(stage_args, stage_config)
    return stage_config


def _require_ckpt(path, fold, stage):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'fold {fold}: {stage} stage produced no checkpoint at {path}')
    return path


def _verify_or_write_fold_manifest(fold_experiment_path, fold_config, cv_folds,
                                    cv_seed, fold, resume, logger):
    """Write fold_manifest.csv (this fold's metadata rows + a 'cv_split' column).
    On --resume, verify an existing manifest against the current --seed/--cv_folds/CSV before any
    stage resumes, so a changed split can't silently continue an existing checkpoint."""
    train_meta = fold_config.dataset.train._base_.METADATA_FILE_PATH
    val_meta = fold_config.dataset.val._base_.METADATA_FILE_PATH
    if train_meta != val_meta:
        print_log(f'[main] fold {fold}: train/val METADATA_FILE_PATH differ '
                  f'({train_meta} vs {val_meta}); using the train path for '
                  f'the fold manifest', logger=logger)

    import pandas as pd
    train_df, valid_df = compute_cv_fold_split(train_meta, cv_folds, cv_seed, fold)
    fresh = pd.concat(
        [train_df.assign(cv_split='train'), valid_df.assign(cv_split='valid')],
        ignore_index=True)
    manifest_path = os.path.join(fold_experiment_path, 'fold_manifest.csv')

    if resume and os.path.exists(manifest_path):
        existing = pd.read_csv(manifest_path)
        key = ['File', 'cv_split']
        matches = (existing[key].sort_values(key).reset_index(drop=True)
                   .equals(fresh[key].sort_values(key).reset_index(drop=True)))
        if not matches:
            raise ValueError(
                f"fold {fold}: --resume found an existing fold_manifest.csv at "
                f"{manifest_path} that does NOT match the split computed from "
                f"the current --seed {cv_seed} / --cv_folds {cv_folds} / "
                f"metadata CSV. Resuming would continue training this fold's "
                f"checkpoints under a different data split. Match the original "
                f"run's --seed/--cv_folds, or use a new --exp_name to start over.")
        print_log(f'[main] fold {fold}: fold_manifest.csv matches the current '
                  f'split — safe to resume', logger=logger)
    else:
        fresh.to_csv(manifest_path, index=False)
        print_log(f'[main] fold {fold}: manifest ({len(train_df)} train / '
                  f'{len(valid_df)} valid rows) saved to {manifest_path}', logger=logger)


def run_cross_validation(args, config, logger):
    """K-fold cross-validation on one config.

    The dataset splits the fold (non-test metadata rows partitioned into --cv_folds by a --seed
    permutation; fold k validates on k, trains on the rest). Per-fold outputs go to
    <root>/fold_<k>/, aggregated summaries to <root>. With --pretrain_config each fold first runs
    the SSL stages on its own split -> <root>/fold_<k>/{dvae,pretrain,finetune}/. --resume: every
    fold/stage resumes independently from its own ckpt-last.pth, after a fold_manifest.csv drift check.
    """
    if _is_pretrain(config):
        raise ValueError('--train_mode cv needs a _Classification --config; '
                         'per-fold SSL pretraining is run via --pretrain_config '
                         '(and --dvae_config for PointBERT)')

    root_experiment_path = args.experiment_path
    root_tfboard_path = args.tfboard_path

    ssl_pipeline = getattr(args, 'pretrain_config', None) is not None
    canonical_metadata = None
    if ssl_pipeline:
        canonical_metadata = config.dataset.train._base_.get('METADATA_FILE_PATH')
        if canonical_metadata is None:
            raise ValueError('the SSL CV pipeline needs a BoneDataset finetune '
                             'config (dataset.train._base_.METADATA_FILE_PATH)')
        n_stages = 3 if args.dvae_config else 2

    results = []
    for k in range(args.cv_folds):
        fold_root = os.path.join(root_experiment_path, f'fold_{k}')
        fold_tfboard_root = os.path.join(root_tfboard_path, f'fold_{k}')
        os.makedirs(fold_root, exist_ok=True)

        print_log(f'========== CV fold {k}/{args.cv_folds - 1} '
                  f'(seed {args.seed}) ==========', logger=logger)

        fold_config = copy.deepcopy(config)
        _inject_cv_split(fold_config.dataset, k, args.cv_folds, args.seed)

        # one manifest per fold (same split for every stage); also the --resume drift guard
        _verify_or_write_fold_manifest(fold_root, fold_config, args.cv_folds,
                                       args.seed, k, args.resume, logger)

        pretrain_ckpt = None
        if ssl_pipeline:
            stage = 1
            dvae_ckpt = None
            if args.dvae_config is not None:
                dvae_args = _make_stage_args(
                    args, f'{args.exp_name}_fold{k}_dvae',
                    os.path.join(fold_root, 'dvae'),
                    os.path.join(fold_tfboard_root, 'dvae'))
                _set_stage_resume(dvae_args, args.resume, logger, f'fold {k} dVAE')
                dvae_args.ckpts = None
                dvae_config = _load_stage_config(dvae_args, args.dvae_config)
                _inject_cv_split(dvae_config.dataset, k, args.cv_folds, args.seed,
                                 metadata_path=canonical_metadata)
                print_log(f'---------- fold {k} stage {stage}/{n_stages}: dVAE '
                          f'({args.dvae_config}) ----------', logger=logger)
                run_training(dvae_args, dvae_config, logger)
                dvae_ckpt = _require_ckpt(
                    os.path.join(dvae_args.experiment_path, 'ckpt-last.pth'), k, 'dVAE')
                stage += 1

            pre_args = _make_stage_args(
                args, f'{args.exp_name}_fold{k}_pretrain',
                os.path.join(fold_root, 'pretrain'),
                os.path.join(fold_tfboard_root, 'pretrain'))
            _set_stage_resume(pre_args, args.resume, logger, f'fold {k} pretrain')
            pre_args.ckpts = None
            pre_config = _load_stage_config(pre_args, args.pretrain_config)
            _inject_cv_split(pre_config.dataset, k, args.cv_folds, args.seed,
                             metadata_path=canonical_metadata)
            if dvae_ckpt is not None:
                pre_config.model.dvae_config.ckpt = dvae_ckpt
            print_log(f'---------- fold {k} stage {stage}/{n_stages}: pretraining '
                      f'({args.pretrain_config}) ----------', logger=logger)
            run_training(pre_args, pre_config, logger)
            pretrain_ckpt = _require_ckpt(
                os.path.join(pre_args.experiment_path, 'ckpt-last.pth'), k, 'pretraining')

        if ssl_pipeline:
            fold_args = _make_stage_args(
                args, f'{args.exp_name}_fold{k}',
                os.path.join(fold_root, 'finetune'),
                os.path.join(fold_tfboard_root, 'finetune'))
            _set_stage_resume(fold_args, args.resume, logger, f'fold {k} finetune')
            fold_args.finetune_model = True
            fold_args.scratch_model = False
            fold_args.ckpts = pretrain_ckpt
            print_log(f'---------- fold {k} stage {n_stages}/{n_stages}: finetuning '
                      f'from {pretrain_ckpt} ----------', logger=logger)
        else:
            fold_args = _make_stage_args(
                args, f'{args.exp_name}_fold{k}', fold_root, fold_tfboard_root)
            _set_stage_resume(fold_args, args.resume, logger, f'fold {k} finetune')

        best = run_training(fold_args, fold_config, logger)
        results.append((k, fold_args.experiment_path, best))

    # aggregate scalar metrics + best-epoch inference timing
    rows = []
    for k, fold_path, best in results:
        row = {'fold': k, 'seed': args.seed}
        for f in CV_METRIC_FIELDS:
            row[f] = float(getattr(best, f, float('nan'))) if best is not None else float('nan')
        timing = getattr(best, 'timing', None) if best is not None else None
        for f in CV_TIMING_FIELDS:
            row[f] = float(timing[f]) if timing else float('nan')
        rows.append(row)

    summary_fields = CV_METRIC_FIELDS + CV_TIMING_FIELDS
    summary_path = os.path.join(root_experiment_path, 'cv_summary.csv')
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        stats = df[summary_fields].agg(['mean', 'std'])
        df.to_csv(summary_path, index=False)
        stats.to_csv(summary_path.replace('.csv', '_stats.csv'))
        print_log('========== Cross-validation summary ==========', logger=logger)
        print_log('\n' + df.to_string(index=False), logger=logger)
        print_log('\n' + stats.to_string(), logger=logger)
    except Exception as e:  # keep the run alive even if pandas is unavailable
        print_log(f'[main] could not write CV summary ({e}); raw results: {rows}',
                  logger=logger)
    print_log(f'CV summary saved to {summary_path}', logger=logger)

    # aggregate per-class metrics of each fold's best epoch
    per_fold_class_metrics = []
    for k, fold_path, best in results:
        best_epoch = getattr(best, 'best_epoch', None) if best is not None else None
        class_metrics = _load_best_epoch_class_metrics(fold_path, best_epoch)
        if class_metrics is None:
            print_log(f'[main] fold {k}: no per-class metrics found for its best '
                      f'epoch ({fold_path}/epoch_metrics.json missing or key '
                      f'{best_epoch!r} absent) — excluded from class summary',
                      logger=logger)
        else:
            per_fold_class_metrics.append(class_metrics)

    class_summary_path = os.path.join(root_experiment_path, 'cv_class_metrics_summary.json')
    with open(class_summary_path, 'w') as f:
        json.dump({
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'folds_included': len(per_fold_class_metrics),
            'per_class_mean_metrics': _aggregate_class_metrics(per_fold_class_metrics),
        }, f, indent=4)
    print_log(f'CV per-class metrics summary saved to {class_summary_path}', logger=logger)


def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first (the logger depends on dist info)
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    config = get_config(args, logger=logger)
    _assign_batch_sizes(args, config)
    # full args/config -> log file only (DEBUG); console gets print_run_summary
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    print_run_summary(args, config, logger)
    logger.info(f'Distributed training: {args.distributed}')
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    _apply_fewshot_args(args, config)

    if args.analysis is not None:
        analysis_fn = ANALYSES.get(args.analysis)
        if analysis_fn is None:
            raise ValueError(f'unknown analysis {args.analysis!r}; '
                             f'available: {sorted(ANALYSES)}')
        print_log(f'[main] running analysis: {args.analysis}', logger=logger)
        analysis_fn(args, config)
    elif args.test:
        if _is_gpt(config):
            test_net_GPT(args, config)
        else:
            test_net(args, config)
    elif args.train_mode == 'cv':
        run_cross_validation(args, config, logger)
    else:  # single training run on the config's pre-made ("random") split
        # --dvae_ckpts overrides model.dvae_config.ckpt for a standalone PointBERT pretrain
        # (CV wires its own per-fold dVAE and never reaches here)
        if args.dvae_ckpts is not None:
            dvae_cfg = config.get('model', {}).get('dvae_config')
            if dvae_cfg is None:
                print_log(f'[main] --dvae_ckpts given but {args.config} has no '
                          f'model.dvae_config — ignoring it', logger=logger)
            else:
                print_log(f'[main] dVAE ckpt override (--dvae_ckpts): '
                          f'{dvae_cfg.get("ckpt")!r} -> {args.dvae_ckpts!r}',
                          logger=logger)
                dvae_cfg.ckpt = args.dvae_ckpts
        run_training(args, config, logger)


if __name__ == '__main__':
    main()

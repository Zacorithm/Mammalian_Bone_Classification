#!/usr/bin/env python3
"""Evaluate the best checkpoint of every CV fold of a model.

Walks ``experiments/<LOSS>/<MODEL>*/fold_<k>/`` (SSL runs nest one level deeper as
``fold_<k>/finetune/``), loads each ``ckpt-best.pth``, runs it over ``--split train|val|test`` and writes:
per-fold metrics + timing -> ``<out>/<split>_fold_metrics.csv``; mean/std -> ``<out>/<split>_summary.csv``;
per-class -> ``<out>/<split>_per_class_metrics.json``; predictions -> ``<out>/fold_<k>/<split>_predictions.csv``.

``--split_source``: ``manifest`` (default for train/val when fold_manifest.csv exists) uses the rows that
fold recorded; ``config`` rebuilds the split from the dataset config (--cv_folds/--cv_seed). ``--split test``
is always ``config``. ``--save_embeddings`` dumps penultimate-layer features to ``<out>/embeddings/``;
``--reduce {umap,tsne,both}`` also saves the 2-D projection.

    python test_performance.py --exp_dir experiments/Cross_entropy/DGCNN_scratch --split val --save_embeddings
    python test_performance.py --exp_dir experiments/Cross_entropy/PCT_scratch --split test --metadata_csv <csv>
"""

import argparse
import copy
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import build_dataset_from_cfg
from datasets.BoneDataset import BoneDataset
from tools import builder
from tools.runner_finetune import calculate_comprehensive_metrics, get_model_reference
from utils import misc
from utils.config import cfg_from_yaml_file
from utils.logger import get_root_logger, print_log

# ``--split`` value -> the ``subset`` string BoneDataset expects
SPLIT_TO_SUBSET = {'train': 'train', 'val': 'valid', 'valid': 'valid', 'test': 'test'}
# ``--split`` value -> the ``cv_split`` value in fold_manifest.csv
SPLIT_TO_CVSPLIT = {'train': 'train', 'val': 'valid', 'valid': 'valid'}

SCALAR_METRIC_FIELDS = [
    'loss', 'overall_acc', 'class_acc', 'recall', 'precision', 'f1_score',
    'mean_infer_ms_per_sample', 'mean_infer_s_per_batch',
]


# args
def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp_dir', required=True,
                   help='experiment root holding fold_*/<ckpt_name> '
                        '(or a single <ckpt_name> for a non-CV run)')
    p.add_argument('--split', default='val', choices=['train', 'val', 'valid', 'test'],
                   help="which split to evaluate and report (default: val)")
    p.add_argument('--config', default=None,
                   help='yaml config (default: <exp_dir>/config.yaml)')
    p.add_argument('--ckpt_name', default='ckpt-best.pth',
                   help='checkpoint file inside each fold dir (default: ckpt-best.pth)')
    p.add_argument('--split_source', default='auto',
                   choices=['auto', 'manifest', 'config'],
                   help="how to determine each fold's samples (default: auto -> "
                        "manifest when fold_manifest.csv exists and split != test)")
    p.add_argument('--folds', default=None,
                   help='comma-separated fold indices to restrict to, e.g. "0,2,4"')
    p.add_argument('--cv_folds', type=int, default=None,
                   help='K for the config-source CV split (default: from cv_summary.csv)')
    p.add_argument('--cv_seed', type=int, default=None,
                   help='seed for the config-source CV split (default: from cv_summary.csv)')
    p.add_argument('--metadata_csv', default=None, help='override dataset METADATA_FILE_PATH')
    p.add_argument('--data_path', default=None, help='override dataset DATA_PATH')
    p.add_argument('--save_embeddings', action='store_true',
                   help='dump per-sample feature vectors (.npz/.pkl) for UMAP/t-SNE')
    p.add_argument('--reduce', default='none', choices=['none', 'umap', 'tsne', 'both'],
                   help='also compute + save the 2-D projection now (implies --save_embeddings)')
    p.add_argument('--output_dir', default=None,
                   help='default: <exp_dir>/evaluation/<split>')
    p.add_argument('--batch_size', type=int, default=None, help='default: config.total_bs')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42,
                   help='torch/numpy seed + fallback CV seed (default: 42)')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()
    args.split = 'val' if args.split == 'valid' else args.split
    if args.reduce != 'none':
        args.save_embeddings = True
    return args


# fold / config discovery
def _find_ckpt(fold_root, ckpt_name):
    """The checkpoint may sit directly in fold_<k>/ (scratch runs) or one level
    down in fold_<k>/finetune/ (the SSL pipeline: dvae/ + pretrain/ + finetune/
    per fold). fold_manifest.csv always stays in fold_<k>/ itself."""
    for cand in (fold_root / ckpt_name, fold_root / 'finetune' / ckpt_name):
        if cand.is_file():
            return cand
    return None


def discover_folds(exp_dir, ckpt_name, folds_arg, logger):
    """Return an ordered list of (fold_index_or_None, fold_root_Path, ckpt_Path).

    fold_root is the ``fold_<k>`` directory (where fold_manifest.csv lives);
    ckpt_Path is the resolved checkpoint, which for SSL fine-tuned runs is
    ``fold_<k>/finetune/<ckpt_name>``."""
    root = Path(exp_dir)
    fold_dirs = {}
    for pth in sorted(root.glob('fold_*')):
        m = re.fullmatch(r'fold_(\d+)', pth.name)
        if not m:
            continue
        ckpt = _find_ckpt(pth, ckpt_name)
        if ckpt is not None:
            fold_dirs[int(m.group(1))] = (pth, ckpt)

    if fold_dirs:
        cv_summary = root / 'cv_summary.csv'
        if folds_arg:
            wanted = [int(x) for x in folds_arg.split(',') if x.strip() != '']
        elif cv_summary.is_file():
            try:
                wanted = sorted(set(pd.read_csv(cv_summary)['fold'].astype(int)))
            except Exception:
                wanted = sorted(fold_dirs)
        else:
            wanted = sorted(fold_dirs)
        missing = [k for k in wanted if k not in fold_dirs]
        if missing:
            print_log(f'[folds] requested folds with no {ckpt_name}: {missing} (skipped)',
                      logger=logger)
        folds = [(k, fold_dirs[k][0], fold_dirs[k][1]) for k in wanted if k in fold_dirs]
        if not folds:
            raise SystemExit(f'none of the requested folds have a {ckpt_name}')
        print_log(f'[folds] evaluating folds {[k for k, _, _ in folds]} under {root}',
                  logger=logger)
        for k, _, ckpt in folds:
            print_log(f'  fold_{k}: {ckpt}', logger=logger)
        return folds

    ckpt = _find_ckpt(root, ckpt_name)
    if ckpt is not None:
        print_log(f'[folds] no fold_*/ dirs — single checkpoint {ckpt}', logger=logger)
        return [(None, root, ckpt)]

    raise SystemExit(
        f'no {ckpt_name} found under {exp_dir} (looked for fold_*/{ckpt_name}, '
        f'fold_*/finetune/{ckpt_name} and {ckpt_name})')


def resolve_cv_params(exp_dir, args, n_folds, logger):
    cv_folds, cv_seed = args.cv_folds, args.cv_seed
    cv_summary = Path(exp_dir) / 'cv_summary.csv'
    if cv_summary.is_file():
        try:
            df = pd.read_csv(cv_summary)
            if cv_seed is None and 'seed' in df.columns and len(df):
                cv_seed = int(df['seed'].iloc[0])
            if cv_folds is None and 'fold' in df.columns and len(df):
                cv_folds = int(df['fold'].max()) + 1
        except Exception as e:
            print_log(f'[cv] could not read {cv_summary}: {e}', logger=logger)
    if cv_seed is None:
        cv_seed = args.seed
    if cv_folds is None:
        cv_folds = n_folds if n_folds else 1
    print_log(f'[cv] config-source split params: cv_folds={cv_folds}, cv_seed={cv_seed}',
              logger=logger)
    return cv_folds, cv_seed


def load_config(args, logger):
    cfg_path = args.config or os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.isfile(cfg_path):
        raise SystemExit(f'config not found: {cfg_path} — pass --config explicitly')
    print_log(f'[config] {cfg_path}', logger=logger)
    return cfg_from_yaml_file(cfg_path)


# dataset building
def _template_dataset_cfg(config):
    """The dataset.<split> block we clone from — they all share the same _base_."""
    for key in ('val', 'train', 'test'):
        if config.dataset.get(key) is not None:
            return config.dataset[key]
    raise SystemExit('config.dataset has no train/val/test block')


def _make_bone_dataset(datapath, list_labels, list_classes, classes, npoints,
                       sampling_algo, data_path):
    """A BoneDataset wired straight from explicit lists — bypasses
    ``BoneDataset.__init__``'s metadata-CSV read entirely (that CSV may be
    missing or stale; the fold_manifest.csv is the split of record). Only the
    attributes ``_get_item`` / ``__getitem__`` / ``__len__`` / ``get_classes``
    / ``get_weights`` touch are set."""
    ds = BoneDataset.__new__(BoneDataset)
    torch.utils.data.Dataset.__init__(ds)
    ds.datapath = list(datapath)
    ds.list_labels = list(list_labels)
    ds.list_classes = list(list_classes)
    ds.classes = classes
    ds.npoints = int(npoints)
    ds.sampling_algo = sampling_algo
    ds.subset = 'valid'          # eval semantics: no train-time point shuffle
    ds.num_category = len(classes)
    ds.data_dir_path = data_path
    ds.metadata_file_path = None
    ds.cv_fold = None
    return ds


def build_config_dataset(config, split, fold, cv_folds, cv_seed, args, logger):
    tmpl = _template_dataset_cfg(config)
    base = copy.deepcopy(tmpl._base_)
    others = copy.deepcopy(tmpl.others)
    others.subset = SPLIT_TO_SUBSET[split]
    if args.metadata_csv:
        others.METADATA_FILE_PATH = args.metadata_csv
    if args.data_path:
        others.DATA_PATH = args.data_path
    if fold is not None and split in ('train', 'val'):
        others.cv_fold = fold
        others.cv_folds = cv_folds
        others.cv_seed = cv_seed
        print_log(f'  [data] config split: subset={others.subset}, '
                  f'cv_fold={fold}/{cv_folds - 1}, cv_seed={cv_seed}', logger=logger)
    else:
        print_log(f'  [data] config split: subset={others.subset} (no CV injection)',
                  logger=logger)
    meta = args.metadata_csv or base.get('METADATA_FILE_PATH')
    if meta and not os.path.isfile(meta):
        raise SystemExit(
            f"config-source split needs a readable metadata CSV, but "
            f"'{meta}' (from the config) does not exist. Pass --metadata_csv "
            f"<path>, or use --split_source manifest (train/val only).")
    dataset = build_dataset_from_cfg(base, others)
    if len(dataset) == 0:
        print_log('  [data] WARNING: 0 samples for this split/config', logger=logger)
    return dataset


def build_manifest_dataset(config, split, fold_dir, args, logger):
    """Evaluate on exactly the rows fold_manifest.csv recorded for this split.

    Reads only fold_manifest.csv (Bone / File / cv_split) + the config's
    DATA_PATH / N_POINTS — never the configured METADATA CSV, which for the
    trained folds has since drifted or been removed."""
    manifest = Path(fold_dir) / 'fold_manifest.csv'
    mdf = pd.read_csv(manifest)
    target = SPLIT_TO_CVSPLIT[split]
    col = mdf['cv_split'].astype(str).str.strip().str.lower()
    sub = mdf[col == target].reset_index(drop=True)

    tmpl = _template_dataset_cfg(config)
    base = copy.deepcopy(tmpl._base_)
    others = copy.deepcopy(tmpl.others)
    npoints = int(others.get('npoints') or base.get('N_POINTS') or config.npoints)
    sampling_algo = str(base.get('sampling_algo', 'fps'))
    data_path = args.data_path or base.get('DATA_PATH')
    if not data_path:
        raise SystemExit('no DATA_PATH in the config and --data_path not given')

    # class map: sorted unique bones over the WHOLE manifest (matches BoneDataset)
    classes = {b: i for i, b in enumerate(sorted(mdf['Bone'].astype(str).unique()))}
    ncat = base.get('NUM_CATEGORY')
    if ncat is not None and len(classes) != int(ncat):
        print_log(f'  [data] WARNING: manifest has {len(classes)} bone classes but '
                  f'config NUM_CATEGORY={ncat}', logger=logger)
    list_classes = list(sub['Bone'].astype(str))
    list_labels = [classes[b] for b in list_classes]
    datapath = [os.path.join(data_path, f) for f in sub['File'].astype(str)]

    dataset = _make_bone_dataset(datapath, list_labels, list_classes, classes,
                                 npoints, sampling_algo, data_path)
    print_log(f'  [data] manifest split: {manifest.parent.name}/fold_manifest.csv '
              f'-> cv_split=={target}: {len(dataset)} samples, {len(classes)} classes '
              f'(npoints={npoints}, DATA_PATH={data_path})', logger=logger)
    return dataset


def make_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, worker_init_fn=misc.worker_init_fn)


# checkpoint loading
def load_checkpoint(base_model, ckpt_path, device, logger):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    if state.get('base_model') is not None:
        raw = state['base_model']
    elif state.get('model') is not None:
        raw = state['model']
    else:
        raw = state
    clean = {k.replace('module.', ''): v for k, v in raw.items()}

    # keep only checkpoint entries the fresh model has AND whose shape matches
    # (tolerates class_weights, PointNet's batch-sized matrix* buffers, etc.);
    # strict=False still raises on a shape mismatch, so filter here
    model_sd = base_model.state_dict()
    filtered, not_in_model, shape_mismatch = {}, [], []
    for k, v in clean.items():
        if k not in model_sd:
            not_in_model.append(k)
        elif hasattr(v, 'shape') and tuple(v.shape) != tuple(model_sd[k].shape):
            shape_mismatch.append(f'{k} {tuple(v.shape)}!={tuple(model_sd[k].shape)}')
        else:
            filtered[k] = v

    incompat = base_model.load_state_dict(filtered, strict=False)
    skipped_names = {s.split(' ')[0] for s in shape_mismatch}
    genuine_missing = [k for k in incompat.missing_keys if k not in skipped_names]

    benign = [k for k in not_in_model if k.split('.')[-1] == 'class_weights']
    other_extra = [k for k in not_in_model if k not in benign]
    if shape_mismatch:
        print_log(f'  [ckpt] skipped {len(shape_mismatch)} shape-mismatched buffer(s): '
                  f'{shape_mismatch[:4]}{" ..." if len(shape_mismatch) > 4 else ""}',
                  logger=logger)
    if other_extra:
        print_log(f'  [ckpt] {len(other_extra)} checkpoint key(s) not in model: '
                  f'{other_extra[:6]}{" ..." if len(other_extra) > 6 else ""}', logger=logger)
    if genuine_missing:
        print_log(f'  [ckpt] WARNING: {len(genuine_missing)} model weight(s) NOT found '
                  f'in checkpoint: {genuine_missing[:6]}'
                  f'{" ..." if len(genuine_missing) > 6 else ""}', logger=logger)
    else:
        print_log(f'  [ckpt] loaded {len(filtered)} tensors, all model weights matched'
                  f'{" (dropped class_weights)" if benign else ""}', logger=logger)
    epoch = state.get('epoch', -1)
    metrics = state.get('metrics', {})
    if not isinstance(metrics, dict) and hasattr(metrics, 'state_dict'):
        metrics = metrics.state_dict()
    return epoch, metrics


# evaluation
@torch.no_grad()
def evaluate(base_model, loader, config, device, want_emb, logger):
    base_model.eval()
    if want_emb:
        base_model.enable_feature_extraction()
    npoints = config.npoints
    model_ref = get_model_reference(base_model)

    preds, labels, embs = [], [], []
    batch_times = []
    total_loss, n_seen, n_emb_missing = 0.0, 0, 0
    loss_ok = True

    for _, _, data in loader:
        points = data[0].to(device, non_blocking=True).contiguous()
        label = data[1].to(device, non_blocking=True).contiguous()
        points = misc.fps(points, npoints).contiguous()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        out = base_model(points)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batch_times.append(time.time() - t0)

        logits = out[0] if isinstance(out, (tuple, list)) else out
        target = label.view(-1)

        if loss_ok:
            try:
                if hasattr(model_ref, 'get_loss_acc'):
                    loss, _ = model_ref.get_loss_acc(logits, target, split='test')
                else:
                    loss = nn.functional.cross_entropy(logits, target)
                total_loss += float(loss) * target.size(0)
            except Exception as e:
                print_log(f'  [eval] loss computation failed ({e}); loss -> nan',
                          logger=logger)
                loss_ok = False

        n_seen += target.size(0)
        preds.append(logits.argmax(-1).view(-1).cpu())
        labels.append(target.cpu())

        if want_emb:
            feat = base_model.get_features_for_visualization()
            if feat is None:
                n_emb_missing += 1
            else:
                if feat.dim() > 2:
                    feat = feat.flatten(2).mean(-1)
                elif feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                embs.append(feat.float().cpu())

    if want_emb:
        base_model.disable_feature_extraction()

    preds = torch.cat(preds).numpy().astype(np.int64)
    labels = torch.cat(labels).numpy().astype(np.int64)
    emb = None
    if want_emb:
        if n_emb_missing:
            print_log(f'  [emb] WARNING: {n_emb_missing} batch(es) returned no '
                      f'features', logger=logger)
        if embs and n_emb_missing == 0:
            emb = torch.cat(embs).numpy().astype(np.float32)
        elif embs:
            print_log('  [emb] partial features discarded (misaligned with samples)',
                      logger=logger)

    return {
        'preds': preds,
        'labels': labels,
        'emb': emb,
        'loss': (total_loss / max(n_seen, 1)) if loss_ok else float('nan'),
        'n': n_seen,
        'batch_times': batch_times,
    }


# output writers
def write_fold_outputs(out_dir, tag, res, dataset, idx2name, num_classes,
                       ckpt_path, ckpt_epoch, ckpt_metrics, split, split_source, logger):
    fold_dir = out_dir / tag
    fold_dir.mkdir(parents=True, exist_ok=True)

    overall_acc, class_acc, recall, precision, f1, per_class = \
        calculate_comprehensive_metrics(
            torch.from_numpy(res['preds']), torch.from_numpy(res['labels']),
            num_classes, idx2name)

    bt = np.asarray(res['batch_times'], dtype=np.float64)
    total_infer = float(bt.sum())
    mean_s_batch = float(bt.mean()) if bt.size else float('nan')
    mean_ms_sample = total_infer / max(res['n'], 1) * 1000.0

    scalars = {
        'fold': tag,
        'split': split,
        'split_source': split_source,
        'n_samples': int(res['n']),
        'ckpt': str(ckpt_path),
        'ckpt_epoch': int(ckpt_epoch) if ckpt_epoch is not None else -1,
        'ckpt_recorded_val_acc': (
            float(ckpt_metrics.get('overall_acc')) if isinstance(ckpt_metrics, dict)
            and ckpt_metrics.get('overall_acc') is not None else float('nan')),
        'loss': float(res['loss']),
        'overall_acc': overall_acc,
        'class_acc': class_acc,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'total_infer_time_s': total_infer,
        'mean_infer_s_per_batch': mean_s_batch,
        'mean_infer_ms_per_sample': mean_ms_sample,
    }

    # per-sample predictions
    paths = list(dataset.datapath)
    if len(paths) != len(res['preds']):
        print_log(f'  [warn] {len(paths)} dataset paths vs {len(res["preds"])} '
                  f'predictions — predictions.csv may be misaligned', logger=logger)
    rows = []
    for i in range(len(res['preds'])):
        pth = paths[i] if i < len(paths) else ''
        lab = int(res['labels'][i])
        prd = int(res['preds'][i])
        rows.append({
            'file_path': pth,
            'basename': os.path.splitext(os.path.basename(str(pth)))[0],
            'label': lab,
            'label_name': idx2name.get(lab, f'class_{lab}'),
            'prediction': prd,
            'prediction_name': idx2name.get(prd, f'class_{prd}'),
            'correct': int(prd == lab),
        })
    pd.DataFrame(rows).to_csv(fold_dir / f'{split}_predictions.csv', index=False)

    with open(fold_dir / f'{split}_metrics.json', 'w') as f:
        json.dump({**scalars, 'per_class': per_class}, f, indent=2)

    with open(fold_dir / f'{split}_inference_time.txt', 'w') as f:
        f.write(f'device / split : {split} ({split_source})\n')
        f.write(f'checkpoint     : {ckpt_path}\n')
        f.write('=' * 60 + '\n')
        f.write(f'samples                     : {res["n"]}\n')
        f.write(f'batches                     : {bt.size}\n')
        f.write(f'total inference time (s)    : {total_infer:.6f}\n')
        f.write(f'mean per batch (s)          : {mean_s_batch:.6f}\n')
        f.write(f'mean per sample (ms)        : {mean_ms_sample:.4f}\n')
        if bt.size:
            f.write(f'min / max per batch (s)     : {bt.min():.6f} / {bt.max():.6f}\n')
            f.write(f'std per batch (s)           : {bt.std():.6f}\n')

    print_log(
        f'  [{tag}] loss={scalars["loss"]:.4f}  OA={overall_acc:.3f}  '
        f'mAcc={class_acc:.3f}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  '
        f'({mean_ms_sample:.2f} ms/sample)', logger=logger)

    return scalars, per_class


def write_embeddings(emb_dir, split, tag, res, dataset, idx2name):
    emb_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in dataset.datapath][:len(res['preds'])]
    files = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    labels, preds = res['labels'], res['preds']
    np.savez_compressed(
        emb_dir / f'{split}_{tag}.npz',
        embeddings=res['emb'],
        labels=labels,
        preds=preds,
        paths=np.asarray(paths, dtype=object),
        files=np.asarray(files, dtype=object),
        label_names=np.asarray([idx2name.get(int(l), f'class_{l}') for l in labels], dtype=object),
        pred_names=np.asarray([idx2name.get(int(p), f'class_{p}') for p in preds], dtype=object),
    )


def write_combined_embeddings(emb_dir, split, per_fold_emb, idx2name, logger):
    feats = np.concatenate([e['emb'] for e in per_fold_emb], axis=0)
    labels = np.concatenate([e['labels'] for e in per_fold_emb], axis=0)
    preds = np.concatenate([e['preds'] for e in per_fold_emb], axis=0)
    paths, folds = [], []
    for e in per_fold_emb:
        p = [str(x) for x in e['paths']]
        paths.extend(p)
        folds.extend([e['fold']] * len(p))
    files = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    label_names = [idx2name.get(int(l), f'class_{l}') for l in labels]

    np.savez_compressed(
        emb_dir / f'{split}_all_folds.npz',
        embeddings=feats, labels=labels, preds=preds,
        fold=np.asarray(folds, dtype=object),
        paths=np.asarray(paths, dtype=object),
        files=np.asarray(files, dtype=object),
        label_names=np.asarray(label_names, dtype=object),
    )
    with open(emb_dir / f'{split}_all_folds.pkl', 'wb') as f:
        pickle.dump({
            'features': feats,
            'labels': labels,
            'preds': preds,
            'fold': np.asarray(folds, dtype=object),
            'paths': np.asarray(paths, dtype=object),
            'files': np.asarray(files, dtype=object),
            'label_names': np.asarray(label_names, dtype=object),
            'class_names': idx2name,
        }, f)
    print_log(f'[emb] pooled embeddings: {feats.shape} -> '
              f'{emb_dir / (split + "_all_folds.npz")}', logger=logger)


def maybe_reduce(emb_dir, split, tag, emb, labels, idx2name, methods, logger):
    try:
        from tools.visualisation import create_visualization
    except Exception as e:
        print_log(f'[reduce] cannot import create_visualization ({e}); skipped',
                  logger=logger)
        return
    if emb is None or len(emb) < 5:
        print_log(f'[reduce] {tag}: too few samples ({0 if emb is None else len(emb)})',
                  logger=logger)
        return
    for method in methods:
        out = emb_dir / f'{split}_{tag}_{method}.png'
        try:
            create_visualization(emb, labels, method=method, save_path=str(out),
                                 class_names=idx2name)
            print_log(f'[reduce] {method}: {out}', logger=logger)
        except Exception as e:
            print_log(f'[reduce] {method} failed for {tag}: {e}', logger=logger)


# main
def main():
    args = get_args()
    misc.set_random_seed(args.seed)

    device = torch.device(
        'cuda' if (args.device in ('auto', 'cuda') and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.exp_dir) / 'evaluation' / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(log_file=str(out_dir / f'eval_{args.split}_{ts}.log'),
                             name='fold_eval')
    print_log(f'== fold evaluation ==  exp_dir={args.exp_dir}  split={args.split}  '
              f'device={device}', logger=logger)

    config = load_config(args, logger)
    is_gpt = str(config.model.NAME).startswith('PointGPT')
    batch_size = args.batch_size or int(config.total_bs)
    print_log(f'[config] model={config.model.NAME}  npoints={config.npoints}  '
              f'batch_size={batch_size}  gpt={is_gpt}', logger=logger)

    folds = discover_folds(args.exp_dir, args.ckpt_name, args.folds, logger)
    cv_folds, cv_seed = resolve_cv_params(args.exp_dir, args, len(folds), logger)

    want_emb = args.save_embeddings
    reduce_methods = {'umap': ['umap'], 'tsne': ['tsne'], 'both': ['umap', 'tsne'],
                      'none': []}[args.reduce]

    fold_rows, per_class_by_fold, per_fold_emb = [], {}, []
    idx2name, num_classes = None, None

    for fold_idx, fold_dir, ckpt_path in folds:
        tag = 'fold_single' if fold_idx is None else f'fold_{fold_idx}'
        print_log(f'---------- {tag} ----------', logger=logger)

        manifest_ok = (Path(fold_dir) / 'fold_manifest.csv').is_file()
        if args.split == 'test':
            source = 'config'
        elif args.split_source == 'manifest':
            source = 'manifest'
        elif args.split_source == 'config':
            source = 'config'
        else:  # auto
            source = 'manifest' if manifest_ok else 'config'
        if source == 'manifest' and not manifest_ok:
            print_log('  [data] no fold_manifest.csv — falling back to config source',
                      logger=logger)
            source = 'config'

        if source == 'manifest':
            dataset = build_manifest_dataset(config, args.split, fold_dir, args, logger)
        else:
            dataset = build_config_dataset(config, args.split, fold_idx,
                                           cv_folds, cv_seed, args, logger)
        if len(dataset) == 0:
            print_log(f'  [skip] {tag}: empty split', logger=logger)
            continue

        if idx2name is None:
            classes = dataset.get_classes()
            idx2name = {v: k for k, v in classes.items()}
            num_classes = len(classes)

        loader = make_loader(dataset, batch_size, args.num_workers)

        base_model = builder.model_builder(config.model, logger=logger)
        if hasattr(base_model, 'disable_feature_extraction'):
            base_model.disable_feature_extraction()
        base_model.to(device)
        ckpt_epoch, ckpt_metrics = load_checkpoint(base_model, ckpt_path, device, logger)

        res = evaluate(base_model, loader, config, device, want_emb, logger)

        scalars, per_class = write_fold_outputs(
            out_dir, tag, res, dataset, idx2name, num_classes,
            ckpt_path, ckpt_epoch, ckpt_metrics, args.split, source, logger)
        fold_rows.append(scalars)
        per_class_by_fold[tag] = per_class

        if want_emb and res['emb'] is not None:
            emb_dir = out_dir / 'embeddings'
            write_embeddings(emb_dir, args.split, tag, res, dataset, idx2name)
            per_fold_emb.append({
                'fold': tag,
                'emb': res['emb'],
                'labels': res['labels'],
                'preds': res['preds'],
                'paths': [str(p) for p in dataset.datapath][:len(res['preds'])],
            })
            maybe_reduce(emb_dir, args.split, tag, res['emb'], res['labels'],
                         idx2name, reduce_methods, logger)

        del base_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not fold_rows:
        raise SystemExit('no fold produced results')

    # aggregate
    df = pd.DataFrame(fold_rows)
    df.to_csv(out_dir / f'{args.split}_fold_metrics.csv', index=False)
    present = [c for c in SCALAR_METRIC_FIELDS if c in df.columns]
    stats = df[present].agg(['mean', 'std'])
    stats.to_csv(out_dir / f'{args.split}_summary.csv')

    class_summary = {}
    all_class_keys = sorted({k for pc in per_class_by_fold.values() for k in pc})
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'support']
    for ck in all_class_keys:
        vals = [pc[ck] for pc in per_class_by_fold.values() if ck in pc]
        class_summary[ck] = {
            mk: float(np.mean([v[mk] for v in vals if mk in v])) if vals else float('nan')
            for mk in metric_keys
        }
        class_summary[ck]['folds_used'] = len(vals)
    with open(out_dir / f'{args.split}_per_class_metrics.json', 'w') as f:
        json.dump({'per_fold': per_class_by_fold, 'mean': class_summary}, f, indent=2)

    if want_emb and per_fold_emb:
        emb_dir = out_dir / 'embeddings'
        with open(emb_dir / f'{args.split}_class_names.json', 'w') as f:
            json.dump(idx2name, f, indent=2)
        if len(per_fold_emb) > 1:
            write_combined_embeddings(emb_dir, args.split, per_fold_emb, idx2name, logger)
            pooled = np.concatenate([e['emb'] for e in per_fold_emb], axis=0)
            pooled_lab = np.concatenate([e['labels'] for e in per_fold_emb], axis=0)
            maybe_reduce(emb_dir, args.split, 'all_folds', pooled, pooled_lab,
                         idx2name, reduce_methods, logger)

    print_log('========== summary (mean over folds) ==========', logger=logger)
    print_log('\n' + df[['fold'] + present].to_string(index=False), logger=logger)
    print_log('\n' + stats.to_string(), logger=logger)
    print_log(f'\nresults written under {out_dir}', logger=logger)


if __name__ == '__main__':
    main()

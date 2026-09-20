import os
import argparse
from pathlib import Path

# Analyses runnable via `--analysis <name>`. To add a new analysis:
#   1. add its name here,
#   2. register a callable (args, config) -> None in ANALYSES inside main.py.
ANALYSIS_CHOICES = ['umap_tsne', 'taxonomy']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help='vote acc')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='autoresume training (interrupted by accident). Also works with '
             '--train_mode cv: every fold/stage independently resumes from its '
             'own ckpt-last.pth if one exists (already-finished folds/stages '
             'do ~0 epochs and move on; a partial one continues; a fold that '
             'never started trains fresh)')

    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model',
        action='store_true',
        default=False,
        help='finetune a classifier from a pretrained checkpoint (--ckpts)')
    parser.add_argument(
        '--scratch_model',
        action='store_true',
        default=False,
        help='train a classifier from scratch')

    parser.add_argument(
        '--train_mode',
        choices=['random', 'cv'],
        default='random',
        help="'random': single run on the CSV's pre-made train/valid split; "
             "'cv': K-fold cross-validation — the dataset pools every non-test "
             "row of the metadata CSV, randomly splits it into --cv_folds "
             "folds using --seed, trains once per fold, and aggregates the "
             "best validation metrics")
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=3,
        help='number of folds for --train_mode cv (the CSV is split at runtime; '
             'no per-fold configs needed)')
    parser.add_argument(
        '--pretrain_config',
        type=str,
        default=None,
        help='SSL pipeline for --train_mode cv: pretraining config run inside '
             'every fold before finetuning, on the same fold split; the '
             'finetune stage then loads that fold\'s pretrain ckpt-last.pth. '
             '--config stays the finetune (classification) config')
    parser.add_argument(
        '--dvae_config',
        type=str,
        default=None,
        help='PointBERT only: dVAE training config run as the first stage of '
             'every fold (before --pretrain_config); its ckpt-last.pth is '
             'wired into the pretrain config\'s model.dvae_config.ckpt')
    parser.add_argument(
        '--dvae_ckpts',
        type=str,
        default=None,
        help='standalone PointBERT pretraining only: path to a trained dVAE '
             'checkpoint to load, overriding the pretrain config\'s '
             'model.dvae_config.ckpt. When omitted, the config value is used. '
             'Rejected with --train_mode cv, which trains and wires its own '
             'per-fold dVAE (via --dvae_config)')

    parser.add_argument(
        '--analysis',
        choices=ANALYSIS_CHOICES,
        default=None,
        help='run an analysis instead of train/test '
             '(umap_tsne: UMAP + t-SNE feature visualization; taxonomy: per-bone '
             'taxonomy visualization). Extensible via ANALYSES in main.py')
    parser.add_argument(
        '--visualisation',
        action='store_true',
        default=False,
        help='alias for --analysis umap_tsne')
    parser.add_argument(
        '--vis_taxonomy',
        action='store_true',
        default=False,
        help='alias for --analysis taxonomy')
    parser.add_argument('--vis_bone', type=str, default='Humerus', help='Bone name for taxonomy visualization')
    parser.add_argument('--vis_split', type=str, default='val', help='Split for taxonomy visualization (train/val)')

    # ---- shapenet / few-shot extras ----------------------------------------
    parser.add_argument(
        '--mode',
        choices=['easy', 'median', 'hard', None],
        default=None,
        help='difficulty mode for shapenet')
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)

    args = parser.parse_args()

    if args.analysis is None:
        if args.visualisation:
            args.analysis = 'umap_tsne'
        elif args.vis_taxonomy:
            args.analysis = 'taxonomy'

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.train_mode == 'cv':
        if args.test or args.analysis is not None:
            raise ValueError(
                '--train_mode cv only applies to training (not --test / --analysis)')
        # --resume IS supported with --train_mode cv: each fold/stage resumes

    if args.dvae_config is not None and args.pretrain_config is None:
        raise ValueError('--dvae_config requires --pretrain_config')
    if args.pretrain_config is not None and args.train_mode != 'cv':
        raise ValueError(
            '--pretrain_config is only supported with --train_mode cv (for a '
            'single random-split run, run the pretrain and finetune stages as '
            'separate commands)')
    if args.dvae_ckpts is not None and args.train_mode == 'cv':
        raise ValueError(
            '--dvae_ckpts is only for a standalone pretrain run; with '
            '--train_mode cv each fold trains and wires its own dVAE '
            '(via --dvae_config)')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode
    args.experiment_path = os.path.join('./experiments', args.exp_name)
    args.tfboard_path = os.path.join('./experiments', 'TFBoard', args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

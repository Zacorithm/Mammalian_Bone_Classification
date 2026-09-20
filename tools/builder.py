import os, sys
import json
import pandas as pd
import torch
import torch.optim as optim
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config, logger=None):
    model = build_model_from_cfg(config)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f'[builder] {config.NAME}: {n_total:,} parameters'
    if n_train != n_total:
        msg += f' ({n_train:,} trainable)'
    print_log(msg, logger=logger)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in (model.module if hasattr(model, 'module') else model).named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                lr_min=1e-6,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)

    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    try:
        optimizer.load_state_dict(state_dict['optimizer'])
        print_log('[RESUME INFO] Successfully loaded optimizer state', logger = logger)
    except ValueError as e:
        print_log(f'[RESUME WARNING] Failed to load optimizer: {e}', logger = logger)
        print_log('[RESUME WARNING] Continuing with fresh optimizer state...', logger = logger)
    except Exception as e:
        print_log(f'[RESUME WARNING] Unexpected error loading optimizer: {e}', logger = logger)
        print_log('[RESUME WARNING] Continuing with fresh optimizer state...', logger = logger)

def load_metrics_history(exp_dir, start_epoch, logger = None):
    """Reload on-disk metrics history (rows before `start_epoch`) so a resumed run appends
    instead of overwriting train/valid_metrics.csv + epoch_metrics.json, and restores the
    "Best ..." trackers. `start_epoch` is the 0-indexed next epoch to run.
    Returns (train_metrics_list, valid_metrics_list, per_class_metrics, best).
    """
    train_metrics_list, valid_metrics_list = [], []
    per_class_metrics = {'train': {}, 'valid': {}}
    best = {
        'best_train_overall_acc': 0., 'best_train_class_acc': 0., 'best_train_recall': 0.,
        'best_train_precision': 0., 'best_train_f1_score': 0.,
        'best_valid_overall_acc': 0., 'best_valid_class_acc': 0., 'best_valid_recall': 0.,
        'best_valid_precision': 0., 'best_valid_f1_score': 0., 'best_epoch': 0,
    }
    if start_epoch <= 0:
        return train_metrics_list, valid_metrics_list, per_class_metrics, best

    train_csv = os.path.join(exp_dir, 'train_metrics.csv')
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        df = df[df['Epoch'] <= start_epoch]
        train_metrics_list = df.to_dict('records')
        if len(df):
            best['best_train_overall_acc'] = float(df['Best Overall Accuracy'].iloc[-1])
            best['best_train_class_acc'] = float(df['Best Class Accuracy'].iloc[-1])
            best['best_train_recall'] = float(df['Best Recall'].iloc[-1])
            best['best_train_precision'] = float(df['Best Precision'].iloc[-1])
            best['best_train_f1_score'] = float(df['Best F1-score'].iloc[-1])

    valid_csv = os.path.join(exp_dir, 'valid_metrics.csv')
    if os.path.exists(valid_csv):
        df = pd.read_csv(valid_csv)
        df = df[df['Epoch'] <= start_epoch]
        valid_metrics_list = df.to_dict('records')
        if len(df):
            best['best_valid_overall_acc'] = float(df['Best Overall Accuracy'].iloc[-1])
            best['best_valid_class_acc'] = float(df['Best Class Accuracy'].iloc[-1])
            best['best_valid_recall'] = float(df['Best Recall'].iloc[-1])
            best['best_valid_precision'] = float(df['Best Precision'].iloc[-1])
            best['best_valid_f1_score'] = float(df['Best F1-score'].iloc[-1])
            best_row = df.loc[df['Overall Accuracy'].idxmax()]
            best['best_epoch'] = int(best_row['Epoch'])

    json_path = os.path.join(exp_dir, 'epoch_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            saved = json.load(f)
        for split in ('train', 'valid'):
            per_class_metrics[split] = {
                k: v for k, v in saved.get(split, {}).items() if int(k) < start_epoch
            }

    print_log(
        f'[RESUME INFO] reloaded metrics history: {len(train_metrics_list)} train row(s), '
        f'{len(valid_metrics_list)} valid row(s) up to epoch {start_epoch}',
        logger = logger,
    )
    return train_metrics_list, valid_metrics_list, per_class_metrics, best

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 
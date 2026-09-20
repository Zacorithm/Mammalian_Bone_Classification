import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tools import builder
from utils import misc, dist_utils
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pickle
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
import pandas as pd
import json
from datasets import data_transforms
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        data_transforms.PointcloudJitter(),
    ]
)

test_transforms = transforms.Compose(
    [
    ]
)

class ComprehensiveMetrics:
    def __init__(self, overall_acc=0., class_acc=0., recall=0., precision=0., f1_score=0.):
        self.overall_acc = overall_acc
        self.class_acc = class_acc
        self.recall = recall
        self.precision = precision
        self.f1_score = f1_score
        self.timing = None      # set to the winning epoch's validate_comprehensive() timing dict
        self.best_epoch = None  # set to the raw (0-indexed) epoch this instance came from

    def better_than(self, other):
        return self.overall_acc > other.overall_acc
    
    def state_dict(self):
        return {
            'overall_acc': self.overall_acc,
            'class_acc': self.class_acc,
            'recall': self.recall,
            'precision': self.precision,
            'f1_score': self.f1_score
        }

def calculate_comprehensive_metrics(predictions, targets, num_classes, class_names=None):
    
    predictions = predictions.long()
    targets = targets.long()
    
    correct = (predictions == targets).float()
    overall_acc = correct.mean() * 100
    
    precision = torch.zeros(num_classes, device=predictions.device)
    recall = torch.zeros(num_classes, device=predictions.device)
    f1 = torch.zeros(num_classes, device=predictions.device)
    support = torch.zeros(num_classes, device=predictions.device)
    class_acc_per_class = torch.zeros(num_classes, device=predictions.device)
    
    confusion_matrix = torch.zeros(num_classes, num_classes, device=predictions.device, dtype=torch.long)
    for t, p in zip(targets.view(-1), predictions.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    
    total_samples = confusion_matrix.sum().float()
    
    for i in range(num_classes):
        # True positives, false positives, false negatives, true negatives
        tp = confusion_matrix[i, i].float()
        fp = confusion_matrix[:, i][torch.arange(num_classes) != i].sum().float()
        fn = confusion_matrix[i, :][torch.arange(num_classes) != i].sum().float()
        tn = total_samples - tp - fp - fn
        
        support[i] = confusion_matrix[i, :].sum()
        
        # Precision: TP / (TP + FP)
        if (tp + fp) > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0
        
        # Recall: TP / (TP + FN)
        if (tp + fn) > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0.0
        
        # F1 score: 2 * (precision * recall) / (precision + recall)
        if (precision[i] + recall[i]) > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0
        
    class_acc_per_class = confusion_matrix.diag().float() / confusion_matrix.sum(1).float().clamp(min=1) * 100
    # Only average over classes that have support > 0 to avoid division issues
    valid_classes = support > 0
    if valid_classes.sum() > 0:
        class_acc = class_acc_per_class[valid_classes].mean()
    else:
        class_acc = torch.tensor(0.0)
    avg_precision = precision.mean() * 100
    avg_recall = recall.mean() * 100
    avg_f1 = f1.mean() * 100
    per_class_metrics = {}
    for i in range(num_classes):
        class_key = class_names.get(i, f'class_{i}') if class_names else i
        per_class_metrics[class_key] = {
            'class_index': i,
            'accuracy': float(class_acc_per_class[i]),
            'precision': float(precision[i] * 100),
            'recall': float(recall[i] * 100),
            'f1_score': float(f1[i] * 100),
            'support': int(support[i])
        }
    
    return float(overall_acc), float(class_acc), float(avg_recall), float(avg_precision), float(avg_f1), per_class_metrics

def get_class_weights_from_dataset(train_dataloader, logger=None):
    if hasattr(train_dataloader.dataset, 'get_weights'):
        try:
            class_weights = train_dataloader.dataset.get_weights()
            print_log(f'Class weights obtained from dataset (Class 0 weight : {class_weights[0]})', logger=logger)
            return class_weights
        except Exception as e:
            print_log(f'Failed to get class weights from dataset: {e}', logger=logger)
            return None
    else:
        print_log('Dataset does not have get_weights method', logger=logger)
        return None

def get_model_reference(base_model):
    if hasattr(base_model, 'module'):
        return base_model.module
    return base_model

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    exp_dir = args.experiment_path if hasattr(args, 'experiment_path') else './experiments'
    import os
    os.makedirs(exp_dir, exist_ok=True)
    
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    
    class_names = None
    if hasattr(train_dataloader.dataset, 'get_classes'):
        classes_dict = train_dataloader.dataset.get_classes()
        class_names = {v: k for k, v in classes_dict.items()}
        num_classes = len(classes_dict)
        print_log(f'Found {num_classes} classes: {list(classes_dict.keys())}', logger=logger)
    else:
        num_classes = getattr(config.dataset, 'num_classes', 40)
        print_log(f'Using config num_classes: {num_classes}', logger=logger)
    
    class_weights = get_class_weights_from_dataset(train_dataloader, logger)
    
    base_model = builder.model_builder(config.model, logger=logger)
    base_model.disable_feature_extraction()
    
    if hasattr(base_model, 'set_class_weights'):
        base_model.set_class_weights(class_weights)
        print_log('Class weights set in model', logger=logger)
    else:
        print_log('Model does not support class weights', logger=logger)
    
    start_epoch = 0
    best_metrics = ComprehensiveMetrics(0., 0., 0., 0., 0.)
    best_metrics_vote = ComprehensiveMetrics(0., 0., 0., 0., 0.)
    metrics = ComprehensiveMetrics(0., 0., 0., 0., 0.)
    
    train_metrics_list = []
    valid_metrics_list = []
    per_class_metrics = {'train': {}, 'valid': {}}
    
    best_train_overall_acc = 0.
    best_train_class_acc = 0.
    best_train_recall = 0.
    best_train_precision = 0.
    best_train_f1_score = 0.
    
    best_valid_overall_acc = 0.
    best_valid_class_acc = 0.
    best_valid_recall = 0.
    best_valid_precision = 0.
    best_valid_f1_score = 0.
    best_epoch = 0

    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = ComprehensiveMetrics(best_metric['overall_acc'], best_metric['class_acc'], best_metric['recall'], best_metric['precision'], best_metric['f1_score'])
        # reload prior epochs' metrics so the resumed run appends to
        # train/valid_metrics.csv and epoch_metrics.json instead of
        # overwriting them with only the epochs run after the resume
        train_metrics_list, valid_metrics_list, per_class_metrics, resumed_best = \
            builder.load_metrics_history(exp_dir, start_epoch, logger = logger)
        best_train_overall_acc = resumed_best['best_train_overall_acc']
        best_train_class_acc = resumed_best['best_train_class_acc']
        best_train_recall = resumed_best['best_train_recall']
        best_train_precision = resumed_best['best_train_precision']
        best_train_f1_score = resumed_best['best_train_f1_score']
        best_valid_overall_acc = resumed_best['best_valid_overall_acc']
        best_valid_class_acc = resumed_best['best_valid_class_acc']
        best_valid_recall = resumed_best['best_valid_recall']
        best_valid_precision = resumed_best['best_valid_precision']
        best_valid_f1_score = resumed_best['best_valid_f1_score']
        best_epoch = resumed_best['best_epoch']
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    
    # DDP vs DataParallel
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        if torch.cuda.device_count() > 1:
            print_log(f'Using DataParallel with {torch.cuda.device_count()} GPUs...', logger = logger)
            os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
            base_model = nn.DataParallel(base_model)
        base_model = base_model.cuda()
    
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        epoch_train_samples = 0

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda(non_blocking=True)
            label = data[1].cuda(non_blocking=True)
            epoch_train_samples += points.size(0)
            
            points = points.contiguous()
            label = label.contiguous()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)
            points = misc.fps(points, point_all)  # (B, npoint, 3)
            points = points.contiguous()
            
            points = train_transforms(points)
            points = points.contiguous()

            ret = base_model(points)

            model_ref = get_model_reference(base_model)
            
            # Use modified loss function that handles class weights and split type
            if hasattr(model_ref, 'get_loss_acc'):
                loss, acc = model_ref.get_loss_acc(ret, label, split='train')
            else:
                loss, acc = model_ref.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        train_epoch_time = epoch_end_time - epoch_start_time
        mean_train_batch_s = train_epoch_time / max(n_batches, 1)
        mean_train_sample_ms = train_epoch_time / max(epoch_train_samples, 1) * 1000.0

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
            train_writer.add_scalar('Time/Epoch/TrainTime_s', train_epoch_time, epoch)
            train_writer.add_scalar('Time/Epoch/TrainTime_ms_per_sample', mean_train_sample_ms, epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) MeanBatch = %.4f (s) MeanSample = %.2f (ms) Losses = %s lr = %.6f' %
            (epoch, train_epoch_time, mean_train_batch_s, mean_train_sample_ms,
             ['%.4f' % l for l in losses.avg()], optimizer.param_groups[0]['lr']), logger = logger)

        with torch.no_grad():
            _, train_metrics, train_per_class_metrics, train_timing = validate_comprehensive(
                base_model, train_dataloader, epoch, None, args, config,
                logger=logger, mode='train', num_classes=num_classes, class_names=class_names
            )
            
            if train_metrics.overall_acc > best_train_overall_acc:
                best_train_overall_acc = train_metrics.overall_acc
            if train_metrics.class_acc > best_train_class_acc:
                best_train_class_acc = train_metrics.class_acc
            if train_metrics.recall > best_train_recall:
                best_train_recall = train_metrics.recall
            if train_metrics.precision > best_train_precision:
                best_train_precision = train_metrics.precision
            if train_metrics.f1_score > best_train_f1_score:
                best_train_f1_score = train_metrics.f1_score
            
            train_metrics_list.append({
                'Epoch': epoch + 1,
                'Loss': losses.avg(0),
                'Overall Accuracy': train_metrics.overall_acc,
                'Class Accuracy': train_metrics.class_acc,
                'Class Recall': train_metrics.recall,
                'Class Precision': train_metrics.precision,
                'Class F1-score': train_metrics.f1_score,
                'Best Overall Accuracy': best_train_overall_acc,
                'Best Class Accuracy': best_train_class_acc,
                'Best Recall': best_train_recall,
                'Best Precision': best_train_precision,
                'Best F1-score': best_train_f1_score,
                'Train Time (s)': train_epoch_time,
                'Mean Train Batch Time (s)': mean_train_batch_s,
                'Mean Train Sample Time (ms)': mean_train_sample_ms,
                'Mean Inference Time (ms/sample)': train_timing['mean_inference_time_ms_per_sample'],
                'Mean Inference Time (s/batch)': train_timing['mean_inference_time_s_per_batch']
            })
            
            per_class_metrics['train'][str(epoch)] = train_per_class_metrics

        if epoch % args.val_freq == 0 and epoch != 0:
            val_loss, metrics, valid_per_class_metrics, valid_timing = validate_comprehensive(
                base_model, test_dataloader, epoch, val_writer, args, config,
                logger=logger, num_classes=num_classes, class_names=class_names
            )

            if metrics.overall_acc > best_valid_overall_acc:
                best_valid_overall_acc = metrics.overall_acc
                best_epoch = epoch + 1
            if metrics.class_acc > best_valid_class_acc:
                best_valid_class_acc = metrics.class_acc
            if metrics.recall > best_valid_recall:
                best_valid_recall = metrics.recall
            if metrics.precision > best_valid_precision:
                best_valid_precision = metrics.precision
            if metrics.f1_score > best_valid_f1_score:
                best_valid_f1_score = metrics.f1_score
                
            valid_metrics_list.append({
                'Epoch': epoch + 1,
                'Loss': val_loss,
                'Overall Accuracy': metrics.overall_acc,
                'Class Accuracy': metrics.class_acc,
                'Class Recall': metrics.recall,
                'Class Precision': metrics.precision,
                'Class F1-score': metrics.f1_score,
                'Best Overall Accuracy': best_valid_overall_acc,
                'Best Class Accuracy': best_valid_class_acc,
                'Best Recall': best_valid_recall,
                'Best Precision': best_valid_precision,
                'Best F1-score': best_valid_f1_score,
                'Validation Time (s)': valid_timing['total_inference_time_s'],
                'Mean Inference Time (ms/sample)': valid_timing['mean_inference_time_ms_per_sample'],
                'Mean Inference Time (s/batch)': valid_timing['mean_inference_time_s_per_batch']
            })
            
            per_class_metrics['valid'][str(epoch)] = valid_per_class_metrics

            better = metrics.better_than(best_metrics)
            if better:
                metrics.timing = valid_timing
                metrics.best_epoch = epoch
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
                
            if args.vote:
                if metrics.overall_acc > 92.1 or (better and metrics.overall_acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        pd.DataFrame(train_metrics_list).to_csv(f'{exp_dir}/train_metrics.csv', index=False)
        if valid_metrics_list:
            pd.DataFrame(valid_metrics_list).to_csv(f'{exp_dir}/valid_metrics.csv', index=False)
        
        with open(f"{exp_dir}/epoch_metrics.json", "w+") as f:
            json.dump(per_class_metrics, f, indent=4)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)

    # final timing summary (means over all recorded epochs)
    if train_metrics_list:
        print_log('[Timing] mean train epoch time = %.3f s | mean train time = %.2f ms/sample | mean train inference = %.2f ms/sample' % (
            float(np.mean([r['Train Time (s)'] for r in train_metrics_list])),
            float(np.mean([r['Mean Train Sample Time (ms)'] for r in train_metrics_list])),
            float(np.mean([r['Mean Inference Time (ms/sample)'] for r in train_metrics_list]))), logger=logger)
    if valid_metrics_list:
        print_log('[Timing] mean validation inference = %.2f ms/sample (%.4f s/batch)' % (
            float(np.mean([r['Mean Inference Time (ms/sample)'] for r in valid_metrics_list])),
            float(np.mean([r['Mean Inference Time (s/batch)'] for r in valid_metrics_list]))), logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    # returned so main.py can aggregate cross-validation folds
    return best_metrics

def validate_comprehensive(base_model, test_dataloader, epoch, val_writer, args, config,
                          logger=None, mode='valid', num_classes=None, class_names=None):
    base_model.eval()

    test_pred = []
    test_label = []
    total_loss = 0.0
    total_samples = 0
    total_inference_time = 0.0
    inference_batches = 0
    npoints = config.npoints

    if num_classes is None:
        num_classes = getattr(config.dataset, 'num_classes', 40)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda(non_blocking=True)
            label = data[1].cuda(non_blocking=True)
            points = points.contiguous()
            label = label.contiguous()

            points = misc.fps(points, npoints)
            points = points.contiguous()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.time()
            logits = base_model(points)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_inference_time += time.time() - infer_start
            inference_batches += 1

            target = label.view(-1)

            model_ref = get_model_reference(base_model)
            if hasattr(model_ref, 'get_loss_acc'):
                loss, _ = model_ref.get_loss_acc(logits, target, split=mode)
            else:
                loss = nn.functional.cross_entropy(logits, target)
            total_loss += loss.item() * target.size(0)  # Accumulate total loss
            total_samples += target.size(0)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
            
        avg_loss = total_loss / total_samples
        mean_infer_batch_s = total_inference_time / max(inference_batches, 1)
        mean_infer_sample_ms = total_inference_time / max(total_samples, 1) * 1000.0

        overall_acc, class_acc, recall, precision, f1_score, per_class_metrics = calculate_comprehensive_metrics(
            test_pred, test_label, num_classes, class_names
        )

        print_log(f'[{mode.capitalize()}] EPOCH: %d  Loss: %.4f, Overall Acc: %.4f, Class Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f, Mean Infer: %.2f ms/sample (%.4f s/batch)' %
                 (epoch, avg_loss, overall_acc, class_acc, recall, precision, f1_score,
                  mean_infer_sample_ms, mean_infer_batch_s), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar(f'Metric/{mode}_Overall_ACC', overall_acc, epoch)
        val_writer.add_scalar(f'Metric/{mode}_Class_ACC', class_acc, epoch)
        val_writer.add_scalar(f'Metric/{mode}_Recall', recall, epoch)
        val_writer.add_scalar(f'Metric/{mode}_Precision', precision, epoch)
        val_writer.add_scalar(f'Metric/{mode}_F1', f1_score, epoch)
        val_writer.add_scalar(f'Time/{mode}_InferTime_ms_per_sample', mean_infer_sample_ms, epoch)

    timing = {
        'total_inference_time_s': total_inference_time,
        'mean_inference_time_s_per_batch': mean_infer_batch_s,
        'mean_inference_time_ms_per_sample': mean_infer_sample_ms,
    }
    return avg_loss, ComprehensiveMetrics(overall_acc, class_acc, recall, precision, f1_score), per_class_metrics, timing

def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda(non_blocking=True)
            label = data[1].cuda(non_blocking=True)
            
            points_raw = points_raw.contiguous()
            label = label.contiguous()
            
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)
                
            local_pred = []

            for kk in range(times):
                points = misc.fps(points_raw, point_all)  # (B, npoint) # (B, N, 3)
                points = points.contiguous()  # Ensure contiguity
                points = test_transforms(points)
                points = points.contiguous()  # Ensure contiguity after transforms
                
                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return ComprehensiveMetrics(acc, 0., 0., 0., 0.)  # Only overall accuracy for vote

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model, logger=logger)
    base_model.disable_feature_extraction()
    
    class_weights = get_class_weights_from_dataset(test_dataloader, logger)
    if hasattr(base_model, 'set_class_weights'):
        base_model.set_class_weights(class_weights)
    
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    test_paths = []
    total_loss = 0.0
    total_samples = 0
    total_inference_time = 0.0
    inference_times = []  # Store individual inference times
    npoints = config.npoints
    
    class_names = None
    device = None
    if hasattr(test_dataloader.dataset, 'get_classes'):
        classes_dict = test_dataloader.dataset.get_classes()
        class_names = {v: k for k, v in classes_dict.items()}
        num_classes = len(classes_dict)
    else:
        num_classes = getattr(config.dataset, 'num_classes', 40)

    dataset_paths = None
    if hasattr(test_dataloader.dataset, 'datapath'):
        dataset_paths = test_dataloader.dataset.datapath
    
    current_idx = 0

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda(non_blocking=True)
            label = data[1].cuda(non_blocking=True)
            points = points.contiguous()
            label = label.contiguous()
            device = points.device

            batch_size = points.size(0)
            if dataset_paths is not None:
                if isinstance(dataset_paths[0], tuple):
                     # Handle ModelNet case where datapath is (class_name, path)
                     batch_paths = [p[1] for p in dataset_paths[current_idx : current_idx + batch_size]]
                else:
                     batch_paths = dataset_paths[current_idx : current_idx + batch_size]
                test_paths.extend(batch_paths)
            current_idx += batch_size

            points = misc.fps(points, npoints)
            points = points.contiguous()

            if args.use_gpu:
                torch.cuda.synchronize()
            inference_start = time.time()
            
            logits = base_model(points)
            
            if args.use_gpu:
                torch.cuda.synchronize()
            inference_end = time.time()
            
            batch_inference_time = inference_end - inference_start
            total_inference_time += batch_inference_time
            inference_times.append(batch_inference_time)
            
            target = label.view(-1)

            model_ref = get_model_reference(base_model)
            if hasattr(model_ref, 'get_loss_acc'):
                loss, _ = model_ref.get_loss_acc(logits, target, split='test')
            else:
                loss = nn.functional.cross_entropy(logits, target)
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        avg_loss = total_loss / total_samples
        mean_inference_time = total_inference_time / len(test_dataloader)
        mean_inference_time_per_sample_ms = total_inference_time / max(total_samples, 1) * 1000.0

        inference_time_file = os.path.join(args.experiment_path, 'inference_time.txt')
        with open(inference_time_file, 'w') as f:
            f.write(f"device used : {device}\n")
            f.write(f"Inference Time Statistics\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total batches: {len(inference_times)}\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Total inference time: {total_inference_time:.6f} seconds\n")
            f.write(f"Mean inference time per batch: {mean_inference_time:.6f} seconds\n")
            f.write(f"Mean inference time per sample: {mean_inference_time_per_sample_ms:.4f} ms\n")
            f.write(f"Min inference time: {min(inference_times):.6f} seconds\n")
            f.write(f"Max inference time: {max(inference_times):.6f} seconds\n")
            f.write(f"Std deviation: {np.std(inference_times):.6f} seconds\n\n")
            f.write(f"Per-batch inference times:\n")
            f.write(f"-" * 50 + "\n")
            for i, t in enumerate(inference_times):
                f.write(f"Batch {i+1}: {t:.6f} seconds\n")

        print_log(f'[INFO] Inference time saved to {inference_time_file}', logger=logger)

        overall_acc, class_acc, recall, precision, f1_score, per_class_metrics = calculate_comprehensive_metrics(
            test_pred, test_label, num_classes, class_names
        )

        print_log('[TEST] Loss: %.4f, Overall Acc: %.4f, Class Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f, Mean Inference Time: %.4f s/batch (%.2f ms/sample)' %
                 (avg_loss, overall_acc, class_acc, recall, precision, f1_score,
                  mean_inference_time, mean_inference_time_per_sample_ms), logger=logger)

        test_metrics_csv = os.path.join(args.experiment_path, 'test_metrics.csv')
        pd.DataFrame([{
            'Loss': avg_loss,
            'Overall Accuracy': overall_acc,
            'Class Accuracy': class_acc,
            'Class Recall': recall,
            'Class Precision': precision,
            'Class F1-score': f1_score,
            'Total Samples': total_samples,
            'Total Batches': len(inference_times),
            'Total Inference Time (s)': total_inference_time,
            'Mean Inference Time (s/batch)': mean_inference_time,
            'Mean Inference Time (ms/sample)': mean_inference_time_per_sample_ms
        }]).to_csv(test_metrics_csv, index=False)
        print_log(f'[INFO] Test metrics saved to {test_metrics_csv}', logger=logger)

        if test_paths:
            df = pd.DataFrame({
                'file_path': test_paths,
                'prediction': test_pred.cpu().numpy(),
                'label': test_label.cpu().numpy()
            })
            
            if class_names:
                df['prediction_name'] = df['prediction'].map(class_names)
                df['label_name'] = df['label'].map(class_names)
                
            csv_path = os.path.join(args.experiment_path, 'test_predictions.csv')
            df.to_csv(csv_path, index=False)
            print_log(f'[INFO] Test predictions saved to {csv_path}', logger=logger)

        if logger:
            print_log("Per-class metrics:", logger=logger)
            for class_key, metrics in per_class_metrics.items():
                print_log(f"  {class_key}: Acc={metrics['accuracy']:.2f}%, "
                         f"Prec={metrics['precision']:.2f}%, "
                         f"Rec={metrics['recall']:.2f}%, "
                         f"F1={metrics['f1_score']:.2f}%, "
                         f"Support={metrics['support']}", logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

'''print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for vote_time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (vote_time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)'''

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda(non_blocking=True)
            label = data[1].cuda(non_blocking=True)
            points_raw = points_raw.contiguous()
            label = label.contiguous()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            local_pred = []

            for kk in range(times):
                points = misc.fps(points_raw, point_all)  # (B, npoint)  # (B, N, 3)
                points = points.contiguous()
                points = test_transforms(points)
                points = points.contiguous()
                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    
    return acc

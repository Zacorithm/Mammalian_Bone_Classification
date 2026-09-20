"""Shared classification loss for every *_Classification model in this repo.

Single source of truth for the loss that used to be duplicated in each model
file. Two losses only, selected via the model-config key ``cls_loss``:

    cls_loss: ce      # cross entropy (default)
    cls_loss: focal   # multi-class focal loss (Lin et al., 2017)

Both honour the same two modifiers, read from the model config:

  * ``use_class_weights`` (default True) -- when True *and* the runner has
    supplied per-class weights (the imbalanced-dataset weights from
    ``dataset.get_weights()``), they are applied:
      - ``ce``    -> passed as ``weight=`` to ``F.cross_entropy``
      - ``focal`` -> the per-class alpha term; each sample's loss is scaled
        by the weight of its target class.
    Set ``use_class_weights: false`` (or provide no weights) for the
    unweighted loss.

  * ``label_smoothing`` (default 0.0, i.e. off unless the config sets it) --
    when > 0, label smoothing is applied at train time only (disabled for
    validation/test, i.e. ``split != 'train'``):
      - ``ce``    -> ``label_smoothing=`` arg of ``F.cross_entropy`` (torch
        spreads ``eps/C`` over all C classes, target included).
      - ``focal`` -> the target becomes the smoothed one-hot ``q`` (``1-eps``
        at the target, ``eps/(C-1)`` on every other class) and the loss is
        ``-sum_c q_c * (1 - p_c)**gamma * log(p_c)``. This collapses exactly
        to the standard ``-(1 - p_t)**gamma * log(p_t)`` when ``eps == 0``.
    Set ``label_smoothing: 0`` to disable smoothing entirely.

``focal`` additionally reads ``focal_gamma`` (default 2.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import print_log


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017): L = -sum_c q_c (1-p_c)**gamma log(p_c),
    q = one-hot (-> plain focal) or the smoothed one-hot when label_smoothing > 0.
    class_weights scale each sample by its target class's weight. Smoothing is train-only.
    """

    def __init__(self, num_classes, gamma=2.0, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, split='train', class_weights=None):
        target = target.long()
        eps = self.label_smoothing if split == 'train' else 0.0

        log_probs = F.log_softmax(pred, dim=1)
        focal_factor = (1.0 - log_probs.exp()) ** self.gamma      # (B, C)
        per_class = -(focal_factor * log_probs)                   # (B, C)

        if eps > 0.0:
            q = torch.full_like(pred, eps / (self.num_classes - 1))
            q.scatter_(1, target.unsqueeze(1), 1.0 - eps)
            loss = (q * per_class).sum(dim=1)                     # (B,)
        else:
            loss = per_class.gather(1, target.unsqueeze(1)).squeeze(1)

        if class_weights is not None:
            if class_weights.device != target.device:
                class_weights = class_weights.to(target.device)
            loss = loss * class_weights[target]

        return loss.mean()


class ClassificationLossMixin:
    """Shared loss plumbing. Call ``_init_cls_loss(config, num_classes)`` after
    ``nn.Module.__init__``; provides ``set_class_weights`` and ``get_loss_acc(ret, gt, split)``.
    """

    VALID_CLS_LOSSES = ('ce', 'focal')

    def _init_cls_loss(self, config, num_classes):
        self.cls_loss_type = getattr(config, 'cls_loss', 'ce')
        assert self.cls_loss_type in self.VALID_CLS_LOSSES, \
            f"cls_loss must be one of {self.VALID_CLS_LOSSES}, got {self.cls_loss_type!r}"
        self.use_class_weights = getattr(config, 'use_class_weights', True)
        self.label_smoothing = getattr(config, 'label_smoothing', 0.0)

        self.focal_loss = FocalLoss(
            num_classes=num_classes,
            gamma=getattr(config, 'focal_gamma', 2.0),
            label_smoothing=self.label_smoothing,
        )
        # buffer so weights travel with .to(device) / state_dict
        self.register_buffer('class_weights', None)

    def set_class_weights(self, class_weights):
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print_log('Class weights are set', logger=type(self).__name__)
        else:
            self.class_weights = None
            print_log('No class weights provided - using standard loss',
                      logger=type(self).__name__)

    def get_loss_acc(self, ret, gt, split='train'):
        gt = gt.long()
        weights = self.class_weights if self.use_class_weights else None

        if self.cls_loss_type == 'focal':
            loss = self.focal_loss(ret, gt, split=split, class_weights=weights)
        else:  # 'ce'
            if weights is not None and weights.device != ret.device:
                weights = weights.to(ret.device)
            smoothing = self.label_smoothing if split == 'train' else 0.0
            loss = F.cross_entropy(ret, gt, weight=weights, label_smoothing=smoothing)

        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

# Mammalian Bone Classification from 3D Point Clouds

A benchmark of **21 point-cloud deep-learning architectures** for classifying mammalian skeletal elements (14 bone types) from 3D scans. Every model is trained and evaluated through one configuration-driven pipeline, under two regimes:

- **Within-distribution:** stratified *K*-fold cross-validation on a development pool.
- **Cross-taxon generalisation:** one-shot splits in which the validation genera (or families) never appear in training.

> **This repository contains the code, the configuration and the dataset metadata** (specimen table, taxon-disjoint splits and a taxonomy sunburst, in [`metadata/`](metadata/)). The 3D scans themselves (`.ply` point clouds), trained checkpoints and experiment results are not included. See [Data preparation](#data-preparation) for how to plug in the point clouds.

## Contents

[Highlights](#highlights) · [Repository layout](#repository-layout) · [Installation](#installation) · [Data preparation](#data-preparation) · [Quick start](#quick-start) · [Training](#training) · [Evaluation](#evaluation) · [Outputs](#outputs) · [Architectures](#architectures) · [Reproducibility](#reproducibility) · [Extending the code](#extending-the-code) · [Acknowledgements](#acknowledgements)

## Highlights

- **21 architectures** behind one interface (`<Name>_Classification`): point-wise MLP, convolution, graph, transformer / attention and state-space (Mamba) models.
- **Input:** each `.ply` cloud is resampled to 1,024 points by farthest-point sampling (FPS), centred and scaled into the unit sphere. Only XYZ coordinates are used.
- **Splits:** stratified *K*-fold CV computed at run time (no per-fold files), or pre-made taxon-disjoint `train` / `valid` splits (genus- and family-disjoint splits are provided in `metadata/`).
- **Self-supervised pretraining → finetuning** for Point-BERT (with its dVAE stage), Point-MAE, PointGPT-S and Point-FEMAE. In CV mode this runs inside every fold, on that fold's training rows only.
- **Imbalance-aware losses:** class-weighted cross-entropy (with label smoothing) or class-balanced focal loss, selected in the YAML config.
- **Evaluation tooling:** independent re-evaluation of saved checkpoints (metrics, per-class results, per-specimen predictions, penultimate-layer embeddings, UMAP / t-SNE).

## Repository layout

```text
.
├── main.py                 # entry point: training, testing, feature-visualisation analyses
├── test_performance.py     # independent evaluation of saved checkpoints
├── requirements.txt        # pinned dependencies (the environment used for training)
├── cfgs/                   # YAML configs: one folder per architecture + dataset_configs/
├── datasets/               # BoneDataset (+ pretraining variant), transforms, dataset registry
├── metadata/               # dataset metadata, taxon-disjoint splits, taxonomy sunburst (see Data preparation)
├── models/                 # the 21 architectures, vendored bidirectional-Mamba ops (bimamba_ssm/),
│                           #   and the KPConv kernel-point layout (kernel_dispositions/)
├── tools/                  # training / evaluation runners, model & optimiser builders, visualisation
└── utils/                  # config & CLI parsing, logging, registry, losses, checkpoints, feature hooks
```

You create two more folders locally; both are git-ignored so that no point clouds or results end up in version control:

| Folder | Content |
|---|---|
| `data/point_clouds/` | the `.ply` files (configurable, see below) |
| `experiments/` | everything a run writes: logs, checkpoints, metrics, TensorBoard files |

## Installation

Requirements: Linux, Python 3.11, and an NVIDIA GPU with a recent CUDA driver. Training needs a GPU; `test_performance.py` can also run on CPU (`--device cpu`). There are no custom CUDA extensions to compile: the Mamba models use the vendored ops in `models/bimamba_ssm` (PyTorch + Triton).

```bash
conda create -n bonecls python=3.11 -y
conda activate bonecls

pip install -r requirements.txt
```

`requirements.txt` pins the versions of the environment used for training (Python 3.11; PyTorch 2.13 and torchvision 0.28 built for CUDA 12.9). `pip` installs the default PyTorch build for your platform; if you need a specific CUDA build, install `torch` and `torchvision` first as described at <https://pytorch.org/get-started/locally/>, then run the command above.

Optional packages (the first two are listed at the end of `requirements.txt`, `flash-attn` is not):

| Package | Needed for |
|---|---|
| `umap-learn` | UMAP projections (`--reduce umap\|both`, `--analysis umap_tsne`) |
| `tensorboard` | viewing the TensorBoard logs |
| `flash-attn` | picked up automatically by Point Transformer V3 when installed |

Run every command **from the repository root**: config, split and experiment paths are relative to it. Quick check that the installation works:

```bash
python -c "import main, test_performance; print('ok')"
```

## Data preparation

The repository ships the dataset **metadata and splits** in [`metadata/`](metadata/). The **point clouds** (`.ply` files) are not distributed with the code: put them in a folder and point the configs at it.

**1. Point clouds** — a folder of `.ply` files named as in the `File` column of the metadata, each with a `vertex` element holding `x`, `y`, `z`. Clouds with at least 1,024 vertices are downsampled by FPS; smaller ones are sampled with replacement. Point `DATA_PATH` at the folder in the dataset configs (default: `data/point_clouds`):

```bash
sed -i 's#^DATA_PATH:.*#DATA_PATH: /path/to/your/ply_folder#' cfgs/dataset_configs/*.yaml
```

**2. Metadata and splits** — provided in [`metadata/`](metadata/):

| File | Content |
|---|---|
| `metadata.csv` | one row per specimen, without a `Split` column. Used for cross-validation and for self-supervised pretraining |
| `split_by_genus.csv` | `metadata.csv` plus a `Split` column (`train` / `valid`), built so that no genus appears on both sides |
| `split_by_family.csv` | the same for families: no family appears on both sides |
| `sunburst.html` | interactive sunburst of the taxonomy (Order → Family → Genus, each slice sized by its number of specimens). Download it and open it in a web browser: GitHub only shows the source of HTML files |

The two split files are the ones used for the cross-taxon generalisation experiments. Columns of the CSV files:

| Column | Meaning |
|---|---|
| `File` | file name of the `.ply` (unique), relative to `DATA_PATH` |
| `Bone` | classification target (skeletal element) |
| `Order`, `Family`, `Genus` | taxonomic path of the specimen; `Order` and `Family` are also read by `--analysis taxonomy` |
| `Class`, `Specie`, `Dataset` | taxonomic class (always `Mammalia`), species epithet (`sp.` in the split files when unknown), source collection |
| `Split` | split files only: `train` or `valid` |

The training code reads `File` and `Bone`, plus `Split` for one-shot runs. If you build your own split files, `Split` may contain `train` / `valid` (and optionally `test`), case-insensitive; CV runs pool every non-`test` row.

Class indices are the sorted unique `Bone` values of the whole CSV, so the label mapping is identical for every split and fold. The reference setup has 14 skeletal elements (Calcaneus, Cuboid, Femur, Fibula, Humerus, Mandible, Navicular, Pelvis, Radius, Scapula, Skull, Talus, Tibia, Ulna). If your label set differs, update `cls_dim` in the model configs and `NUM_CATEGORY` in `cfgs/dataset_configs/*.yaml`.

Which dataset config reads which CSV:

| Dataset config (`cfgs/dataset_configs/`) | CSV | Used for |
|---|---|---|
| `bone_dataset_config.yaml` | `metadata/metadata.csv` | CV runs (`--train_mode cv`); the CSV has no `Split` column, so every row is pooled |
| `bone_dataset_config_genus.yaml` | `metadata/split_by_genus.csv` | genus-disjoint one-shot runs |
| `bone_dataset_config_family.yaml` | `metadata/split_by_family.csv` | family-disjoint one-shot runs |
| `bone_dataset_pretrain.yaml` | `metadata/metadata.csv` | self-supervised pretraining |

The split files were generated with `StratifiedGroupKFold` (stratified on `Bone`, grouped by the taxonomic rank); of the 5 candidate folds, the one closest to 20 % of the specimens becomes `valid`. The snippet below implements this procedure. By default it builds an order-disjoint split as an example of another rank (`metadata/split_by_order.csv`; to train on a new split file, copy one of the `bone_dataset_config_*.yaml` files and change `METADATA_FILE_PATH`). Setting `RANKS = ["Genus", "Family"]` regenerates the provided files, which it reproduces byte for byte with scikit-learn 1.7.2; other versions (for example 1.8.0) assign the folds differently, so treat the provided CSVs as the reference.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

VALID_FRAC, SEED = 0.20, 42
RANKS = ["Order"]      # ["Genus", "Family"] regenerates the provided files (see above)

df = pd.read_csv("metadata/metadata.csv")
for c in df.select_dtypes("object").columns:      # normalise strings; missing values become the string "nan"
    df[c] = df[c].astype(str).str.strip()
df.loc[df["Specie"].isin(["nan", "None", ""]), "Specie"] = "sp."   # unknown species epithet

def single_grouped_split(frame, target_col, group_col, valid_frac, seed):
    """One StratifiedGroupKFold fold (the one closest to valid_frac in size) becomes 'valid'."""
    k = max(2, round(1.0 / valid_frac))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [val_idx for _, val_idx in sgkf.split(frame, frame[target_col], frame[group_col])]
    best = min(range(k), key=lambda i: abs(len(folds[i]) / len(frame) - valid_frac))
    split = np.full(len(frame), "train", dtype=object)
    split[folds[best]] = "valid"
    return split

for rank in RANKS:
    out = df.assign(Split=single_grouped_split(df, "Bone", rank, VALID_FRAC, SEED))
    out.to_csv(f"metadata/split_by_{rank.lower()}.csv", index=False)
```

The grouping constraint can leave some classes with very few validation specimens, especially for coarse ranks such as `Order`; check the per-class counts of each split.

## Quick start

```bash
# 1. train PointNet with 5-fold cross-validation (outputs go to experiments/demo/PointNet)
python main.py --config cfgs/PointNet/train_BoneDataset.yaml \
    --scratch_model --train_mode cv --cv_folds 5 --seed 42 --exp_name demo/PointNet

# 2. re-evaluate every fold's best checkpoint on its validation split, and dump embeddings
python test_performance.py --exp_dir experiments/demo/PointNet --split val --save_embeddings

# 3. follow the training curves
tensorboard --logdir experiments/TFBoard
```

## Training

All training goes through `main.py`. The main options:

| Option | Meaning |
|---|---|
| `--config` | model YAML; it pulls in its dataset config through `_base_` |
| `--exp_name` | run name; outputs go to `experiments/<exp_name>/` |
| `--scratch_model` / `--finetune_model` | train from scratch / finetune from the checkpoint given by `--ckpts` (`*_Classification` configs default to scratch) |
| `--train_mode {random,cv}` | `random` (default): one run on the CSV's `train` / `valid` split. `cv`: *K*-fold cross-validation |
| `--cv_folds`, `--seed` | number of folds (default 3; the provided scripts use 5) and random seed (default 42) |
| `--pretrain_config`, `--dvae_config` | self-supervised pipeline inside every fold (CV only), see below |
| `--ckpts` | checkpoint to load (finetuning, `--test`, `--analysis`) |
| `--resume` | continue an interrupted run; with CV each fold / stage resumes from its own `ckpt-last.pth` |
| `--test` | evaluate `--ckpts` on the split configured as `dataset.val` |
| `--analysis {umap_tsne,taxonomy}` | feature visualisation (pass `--ckpts` to visualise a trained model) |

### 5-fold CV from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/DGCNN/train_BoneDataset.yaml \
    --scratch_model --train_mode cv --cv_folds 5 --seed 42 --exp_name cv5/DGCNN
```

### Self-supervised pretraining and finetuning in every fold

Architectures with `pretrain_*` / `finetune_*` configs (Point-BERT, Point-MAE, PointGPT-S, Point-FEMAE) run *pretrain → finetune* per fold. `--config` is the finetune (classification) config:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config cfgs/PointMAE/finetune_BoneDataset.yaml \
    --pretrain_config cfgs/PointMAE/pretrain_BoneDataset.yaml \
    --train_mode cv --cv_folds 5 --seed 42 --exp_name cv5/PointMAE

# Point-BERT trains its discrete VAE first:
#   --dvae_config cfgs/PointBERT/train_dvae_BoneDataset.yaml
```

### One-shot run on a pre-made split (taxon-disjoint)

Without `--train_mode cv`, `dataset.train` reads the rows with `Split == train` and `dataset.val` those with `Split == valid`. To use a taxon-disjoint split, point a copy of the model config at the matching dataset config:

```bash
sed 's#bone_dataset_config.yaml#bone_dataset_config_genus.yaml#g' \
    cfgs/PointNet/train_BoneDataset.yaml > cfgs/PointNet/train_BoneDataset_genus.yaml

python main.py --config cfgs/PointNet/train_BoneDataset_genus.yaml \
    --scratch_model --exp_name genus/PointNet
```

Use `_family` for the family-disjoint split. For self-supervised models a one-shot run is two commands (pretrain, then finetune from its checkpoint):

```bash
python main.py --config cfgs/PointMAE/pretrain_BoneDataset.yaml --exp_name genus/PointMAE/pretrain
python main.py --config <finetune config for the split> --finetune_model \
    --ckpts experiments/genus/PointMAE/pretrain/ckpt-last.pth --exp_name genus/PointMAE/finetune
```

> In one-shot mode the pretraining dataset (`BoneDataset_pretrain`) reads **every row** of its `METADATA_PATH` and ignores `Split`. If pretraining must not see the held-out taxa, point `METADATA_PATH` in `cfgs/dataset_configs/bone_dataset_pretrain.yaml` at a CSV that contains only the training rows.

### Loss, class weights and hyper-parameters

They live in the model YAML (`model:` block and top level):

```yaml
model:
  cls_loss: focal          # 'ce' (default) | 'focal'; both honour class weights and label smoothing
  use_class_weights: true  # inverse-frequency ("balanced") weights, recomputed from each training fold
  label_smoothing: 0.1
  focal_gamma: 2.0
max_epoch: 150
total_bs: 64
```

All classification configs use AdamW (lr 1e-3, weight decay 0.05) with a cosine schedule (10-epoch warm-up) for 150 epochs; the pretraining stages use the same optimiser for 300 epochs.

### Several architectures, several GPUs, TensorBoard

- To train several architectures one after the other on one GPU, loop over their configs; start one such loop per GPU to use several GPUs in parallel:

  ```bash
  for m in PointNet PointMLP PointNeXt; do
      CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/$m/train_BoneDataset.yaml \
          --scratch_model --train_mode cv --cv_folds 5 --seed 42 --exp_name cv5/$m
  done
  ```
- If several GPUs are visible to one process (`CUDA_VISIBLE_DEVICES=0,1`), the runners wrap the model in `nn.DataParallel`.
- Training curves are written to `experiments/TFBoard/<exp_name>/` (`tensorboard --logdir experiments/TFBoard`).

## Evaluation

`test_performance.py` reloads the `ckpt-best.pth` of every fold (or of a single one-shot run) and recomputes all metrics independently of the training loop:

```bash
# validation split of every fold; also saves penultimate-layer embeddings
python test_performance.py --exp_dir experiments/cv5/DGCNN --split val --save_embeddings

# additionally render the 2-D UMAP and t-SNE projections
python test_performance.py --exp_dir experiments/cv5/DGCNN --split val --save_embeddings --reduce both

# training / test split, custom metadata or data folder, CPU
python test_performance.py --exp_dir experiments/cv5/DGCNN --split train \
    --metadata_csv metadata/metadata.csv --data_path data/point_clouds --device cpu
```

Results land in `<exp_dir>/evaluation/<split>/`:

| File | Content |
|---|---|
| `<split>_fold_metrics.csv` | per-fold metrics (accuracy, class accuracy, macro recall / precision / F1, loss, inference time) |
| `<split>_summary.csv` | mean and standard deviation over folds |
| `<split>_per_class_metrics.json` | per-class breakdown |
| `fold_<k>/<split>_predictions.csv` | per-specimen predictions |
| `embeddings/` | penultimate-layer features (and projections with `--reduce`) |

For SSL runs the script looks in `fold_<k>/finetune/`. When `fold_manifest.csv` exists, the `train` / `val` rows are taken from it, so the evaluation uses exactly the fold's original membership (`--split_source config` rebuilds the split from the config instead).

**Several runs.** `test_performance.py` evaluates one run folder per call, so loop over your runs. For a one-shot run, `--exp_dir` is the run folder itself (for the self-supervised models, its `finetune/` sub-folder):

```bash
# every CV run under experiments/cv5/
for d in experiments/cv5/*/; do
    python test_performance.py --exp_dir "$d" --split val --save_embeddings
done

# one-shot taxon-disjoint runs, with UMAP / t-SNE projections
python test_performance.py --exp_dir experiments/genus/PointNet --split val --save_embeddings --reduce both
python test_performance.py --exp_dir experiments/genus/PointMAE/finetune --split val --save_embeddings --reduce both
```

**Single checkpoint / visualisation through `main.py`:**

```bash
python main.py --config cfgs/DGCNN/train_BoneDataset.yaml --test --ckpts <path/to/ckpt-best.pth>
python main.py --config cfgs/DGCNN/train_BoneDataset.yaml --analysis umap_tsne --ckpts <path/to/ckpt-best.pth>
python main.py --config cfgs/DGCNN/train_BoneDataset.yaml --analysis taxonomy --vis_bone Humerus --vis_split val --ckpts <path/to/ckpt-best.pth>
```

## Outputs

```text
experiments/<exp_name>/
├── <timestamp>.log                     # full log (args + config)
├── config.yaml                         # copy of the config used
├── cv_summary.csv, cv_summary_stats.csv, cv_class_metrics_summary.json     # CV runs: aggregated over folds
├── fold_<k>/                           # one folder per fold
│   ├── ckpt-best.pth, ckpt-last.pth    #   best epoch (overall accuracy) / latest epoch
│   ├── train_metrics.csv, valid_metrics.csv, epoch_metrics.json
│   ├── fold_manifest.csv               #   exact train / valid membership of this fold
│   └── dvae/, pretrain/, finetune/     #   SSL runs: one sub-folder per stage (files above sit in finetune/)
└── evaluation/<split>/…                # written by test_performance.py
experiments/TFBoard/<exp_name>/         # TensorBoard event files
```

A one-shot (non-CV) run writes the same per-run files directly into `experiments/<exp_name>/`.

## Architectures

| Family | Architecture | Config folder | Implementation |
|---|---|---|---|
| Point-wise MLP | PointNet | `cfgs/PointNet/` | `models/PointNet.py` |
| | PointNet++ (SSG / MSG / MRG) | `cfgs/PointNet2/` | `models/PointNet2.py` |
| | PointMLP (+ ELITE variant) | `cfgs/PointMLP/` | `models/PointMLP.py` |
| | PointNeXt | `cfgs/PointNeXt/` | `models/PointNeXt.py` |
| Convolution | PointCNN | `cfgs/PointCNN/` | `models/PointCNN.py` |
| | PointConv | `cfgs/PointConv/` | `models/PointConv.py` |
| | KPConv | `cfgs/KPConv/` | `models/KPConv.py` |
| | SpiderCNN | `cfgs/SpiderCNN/` | `models/SpiderCNN.py` |
| Graph | DGCNN | `cfgs/DGCNN/` | `models/DGCNN.py` |
| | DeepGCN | `cfgs/DeepGCN/` | `models/DeepGCN.py` |
| | Grid-GCN | `cfgs/GridGCN/` | `models/GridGCN.py` |
| Transformer / attention | PCT | `cfgs/PCT/` | `models/PCT.py` |
| | Point Transformer v1 / v2 / v3 | `cfgs/PointTransformer{,2,3}/` | `models/PointTransformer{,2,3}.py` |
| | Point-BERT † | `cfgs/PointBERT/` | `models/Point_BERT.py` |
| | Point-MAE † | `cfgs/PointMAE/` | `models/Point_MAE.py` |
| | PointGPT-S † | `cfgs/PointGPT-S/` | `models/Point_GPT.py` |
| | Point-FEMAE † | `cfgs/PointFEMAE/` | `models/Point_FEMAE.py` |
| State-space (Mamba) | PointMamba | `cfgs/PointMAMBA/` | `models/Point_MAMBA.py` |
| | Mamba3D | `cfgs/Mamba3D/` | `models/Mamba3D.py` |

† has `pretrain_*` and `finetune_*` configs (self-supervised pretraining, then finetuning); Point-BERT also has `train_dvae_*`. All other architectures have a single `train_BoneDataset.yaml`. Model names in the configs follow `<Name>_Classification` (classifier) or `<Name>_Pretrain` (pretraining).

## Reproducibility

- `--seed` (default 42) seeds Python, NumPy and PyTorch; `--deterministic` additionally sets the deterministic cuDNN options. `cudnn.benchmark` is on by default, so the *data split* is exactly reproducible but the forward / backward computation is not bit-exact.
- CV folds come from `sklearn`'s `StratifiedKFold(shuffle=True, random_state=<seed>)`, stratified on `Bone`, over all rows whose `Split` is not `test`. The same function (`compute_cv_fold_split`) is used by the dataset and by `main.py`.
- Each fold's membership is frozen to `fold_<k>/fold_manifest.csv`. With `--resume`, `main.py` checks the manifest against the split recomputed from the current `--seed`, `--cv_folds` and CSV, and refuses to continue if they differ.
- Class weights come from `sklearn`'s `"balanced"` rule on each training fold when `use_class_weights: true`.

## Extending the code

- **New architecture.** Add `models/<Name>.py` with a class registered by `@MODELS.register_module()` and named `<Name>_Classification`. Use an existing model (for example `models/PointNet.py`) as a template: it calls `self._init_cls_loss(...)` from `ClassificationLossMixin` (`utils/cls_losses.py`), which provides the `ce` / `focal` losses, `set_class_weights` and `get_loss_acc`. Then import the module in `models/__init__.py` and add `cfgs/<Name>/train_BoneDataset.yaml`, copying an existing config and changing `model.NAME`.
- **New dataset.** Register a `Dataset` with `@DATASETS.register_module()` (see `datasets/BoneDataset.py`), import it in `datasets/__init__.py` and reference it from a config under `cfgs/dataset_configs/`.
- **New analysis.** Add a `(args, config) -> None` callable to `ANALYSES` in `main.py` and its name to `ANALYSIS_CHOICES` in `utils/parser.py`.

## Acknowledgements

The training framework (config and registry system, runners) follows the open-source [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) code base (MIT licence). The bidirectional selective-scan ops in `models/bimamba_ssm/` are adapted from the Mamba / Vision Mamba projects; see the copyright headers in those files. The architectures follow their original papers (PointNet, PointNet++, DGCNN, PCT, Point Transformer v1–v3, PointMLP, PointNeXt, PointCNN, PointConv, KPConv, SpiderCNN, DeepGCN, Grid-GCN, Point-BERT, Point-MAE, PointGPT, Point-FEMAE, PointMamba, Mamba3D). When redistributing derived code, keep the upstream copyright and licence notices and check the licences of the projects you reuse.

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import pandas as pd
from tools import builder
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from utils.logger import *
from utils import misc

def extract_features(base_model, dataloader, config, logger=None):
    base_model.eval()
    base_model.enable_feature_extraction()
    
    all_features = []
    all_labels = []
    
    npoints = config.npoints
    samples_processed = 0
    
    try:
        with torch.no_grad():
            for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataloader, desc='Extracting features')):
                    
                points = data[0].cuda(non_blocking=True)
                labels = data[1].cuda(non_blocking=True)
                points = points.contiguous()
                labels = labels.contiguous()
                
                if npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                elif npoints == 4096:
                    point_all = 4800
                elif npoints == 8192:
                    point_all = 8192
                else:
                    point_all = points.size(1)
                
                if points.size(1) < point_all:
                    point_all = points.size(1)
                    
                points = misc.fps(points, point_all)
                points = points.contiguous()
                
                # Forward pass (this will store features in the model)
                _ = base_model(points)
                
                features = base_model.get_features_for_visualization()
                
                if features is not None:
                    if features.dim() > 2:
                        features = torch.mean(features, dim=-1)
                    
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    
                    samples_processed += features.shape[0]
                else:
                    print_log(f"Warning: No features extracted for batch {idx}", logger=logger)
    
    finally:
        base_model.disable_feature_extraction()
    
    if all_features:
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        return features_array, labels_array
    else:
        return None, None

def create_visualization(features, labels, method='umap', 
                        save_path=None, class_names=None, figsize=(12, 8)):
    """features (N, D), labels (N,), method 'umap'|'tsne' -> 2-D scatter saved to save_path."""
    try:
        if method.lower() == 'umap':
            import umap
            reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(features)
        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=50)
            embedding = tsne.fit_transform(features)
        else:
            raise ValueError("Method must be 'umap' or 'tsne'")
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Select appropriate colormap and extract only needed colors
        if n_classes <= 10:
            base_cmap = plt.cm.get_cmap('tab10')
            colors = [base_cmap(i) for i in range(n_classes)]
        elif n_classes <= 20:
            base_cmap = plt.cm.get_cmap('tab20')
            colors = [base_cmap(i) for i in range(n_classes)]
        else:
            # Use seaborn husl palette for distinct colors instead of gradient
            colors = sns.color_palette("hls", n_classes)
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        # Map labels to consecutive indices for proper coloring
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_mapping[label] for label in labels])
            
        plt.figure(figsize=figsize)
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=mapped_labels, cmap=cmap, alpha=0.7, s=20, vmin=0, vmax=n_classes-1)
        plt.xlabel(f'{method.upper()}_1')
        plt.ylabel(f'{method.upper()}_2')
        
        if class_names:
            legend_labels = [class_names.get(int(label), f'Class {label}') for label in unique_labels]
            cbar = plt.colorbar(scatter, ticks=unique_labels)
            cbar.ax.set_yticklabels(legend_labels)
        else:
            plt.colorbar(scatter, label='Classes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
            
    except ImportError as e:
        print(f"Missing required package for {method}: {e}")
        return None
    except Exception as e:
        print(f"Error creating {method} visualization: {e}")
        return None

def visualize_bone_taxonomy(args, config, bone_name, split='val', save_dir=None):
    logger = get_logger(args.log_name)
    
    if save_dir is None:
        exp_dir = args.experiment_path if hasattr(args, 'experiment_path') else './experiments'
        save_dir = os.path.join(exp_dir, 'visualizations', bone_name)
    os.makedirs(save_dir, exist_ok=True)
    
    from datasets import build_dataset_from_cfg
    
    if split == 'all':
        dataset_train = build_dataset_from_cfg(config.dataset.train._base_, config.dataset.train.others)
        dataset_val = build_dataset_from_cfg(config.dataset.val._base_, config.dataset.val.others)
        dataset = ConcatDataset([dataset_train, dataset_val])
        # Use train config for metadata path lookup (assuming shared metadata file)
        dataset_config = config.dataset.train
    elif split == 'train':
        dataset_config = config.dataset.train
        dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
    elif split == 'val':
        dataset_config = config.dataset.val
        dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
    else:
        raise ValueError(f"Unknown split: {split}")

    dataloader = DataLoader(dataset, batch_size=config.total_bs, shuffle=False, num_workers=args.num_workers)
    
    base_model = builder.model_builder(config.model, logger=logger)
    
    if hasattr(args, 'ckpts') and args.ckpts:
        base_model.load_model_from_ckpt(args.ckpts)
        print_log(f"Loaded checkpoint from {args.ckpts}", logger=logger)
    elif hasattr(config, 'ckpt') and config.ckpt:
        base_model.load_model_from_ckpt(config.ckpt)
        print_log(f"Loaded checkpoint from {config.ckpt}", logger=logger)
        
    if torch.cuda.is_available():
        base_model = base_model.cuda()
        
    print_log(f"Extracting features for {split} set...", logger=logger)
    features, _ = extract_features(base_model, dataloader, config, logger=logger)
    
    if features is None:
        print_log("Feature extraction failed.", logger=logger)
        return

    if hasattr(dataset_config, 'METADATA_FILE_PATH'):
        metadata_path = dataset_config.METADATA_FILE_PATH
    elif hasattr(dataset_config, '_base_') and hasattr(dataset_config._base_, 'METADATA_FILE_PATH'):
        metadata_path = dataset_config._base_.METADATA_FILE_PATH
    else:
        print_log("Error: METADATA_FILE_PATH not found in config", logger=logger)
        return

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print_log(f"Error reading metadata file: {e}", logger=logger)
        return
        
    if split == 'all':
        train_subset = config.dataset.train.others.subset.strip().lower()
        val_subset = config.dataset.val.others.subset.strip().lower()
        
        # We must preserve the order: train then val (as ConcatDataset does)
        df_train = df[df['Split'].str.strip().str.lower() == train_subset]
        df_val = df[df['Split'].str.strip().str.lower() == val_subset]
        
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        # Use the subset defined in the config (e.g., 'valid') instead of the arg (e.g., 'val')
        if hasattr(dataset_config, 'others') and hasattr(dataset_config.others, 'subset'):
            target_split = dataset_config.others.subset.strip().lower()
        else:
            target_split = split.strip().lower()
            
        df = df[df['Split'].str.strip().str.lower() == target_split]
        
    df = df.reset_index(drop=True)
    
    if len(df) != len(features):
        print_log(f"Warning: Metadata length ({len(df)}) does not match features length ({len(features)})", logger=logger)
    
    print_log("Enriching metadata with external files...", logger=logger)
    
    def get_basename(path):
        return os.path.splitext(os.path.basename(str(path)))[0]

    df['basename'] = df['File'].apply(get_basename)
    
    # 1. Merge with metadata_v3.csv
    meta_v3_path = 'data/metadata_v3.csv'
    if os.path.exists(meta_v3_path):
        try:
            meta_v3 = pd.read_csv(meta_v3_path)
            if 'File' in meta_v3.columns:
                meta_v3['basename'] = meta_v3['File'].apply(get_basename)
                cols = ['basename']
                if 'Order' in meta_v3.columns: cols.append('Order')
                if 'Family' in meta_v3.columns: cols.append('Family')
                
                meta_v3 = meta_v3[cols].drop_duplicates(subset='basename')
                
                df = pd.merge(df, meta_v3, on='basename', how='left', suffixes=('', '_v3'))
                
                if 'Order_v3' in df.columns:
                    if 'Order' not in df.columns: df['Order'] = df['Order_v3']
                    else: df['Order'] = df['Order'].fillna(df['Order_v3'])
                if 'Family_v3' in df.columns:
                    if 'Family' not in df.columns: df['Family'] = df['Family_v3']
                    else: df['Family'] = df['Family'].fillna(df['Family_v3'])
        except Exception as e:
            print_log(f"Error merging with metadata_v3: {e}", logger=logger)
    else:
        print_log(f"Warning: {meta_v3_path} not found", logger=logger)

    # 2. Merge with merged_data.csv
    merged_data_path = 'data/merged_data.csv'
    if os.path.exists(merged_data_path):
        try:
            merged_data = pd.read_csv(merged_data_path)
            if 'path' in merged_data.columns and 'physical_object_taxonomy_gbif' in merged_data.columns:
                merged_data['basename'] = merged_data['path'].apply(get_basename)
                
                def extract_tax(row):
                    tax = str(row['physical_object_taxonomy_gbif'])
                    parts = [p.strip() for p in tax.split(',')]
                    order = parts[3] if len(parts) > 3 else None
                    family = parts[4] if len(parts) > 4 else None
                    return pd.Series([order, family])
                
                merged_data[['Order_merged', 'Family_merged']] = merged_data.apply(extract_tax, axis=1)
                
                merged_subset = merged_data[['basename', 'Order_merged', 'Family_merged']].drop_duplicates(subset='basename')
                
                df = pd.merge(df, merged_subset, on='basename', how='left')
                
                if 'Order' not in df.columns: df['Order'] = df['Order_merged']
                else: df['Order'] = df['Order'].fillna(df['Order_merged'])
                
                if 'Family' not in df.columns: df['Family'] = df['Family_merged']
                else: df['Family'] = df['Family'].fillna(df['Family_merged'])
        except Exception as e:
            print_log(f"Error merging with merged_data: {e}", logger=logger)
    else:
        print_log(f"Warning: {merged_data_path} not found", logger=logger)
    
    # Normalize Order and Family to Title Case to avoid duplicates
    if 'Order' in df.columns:
        df['Order'] = df['Order'].str.title()
    if 'Family' in df.columns:
        df['Family'] = df['Family'].str.title()

    bone_mask = df['Bone'] == bone_name
    if not bone_mask.any():
        print_log(f"No samples found for bone: {bone_name}", logger=logger)
        return
        
    target_features = features[bone_mask.values]
    target_df = df[bone_mask]
    
    print_log(f"Found {len(target_features)} samples for {bone_name}", logger=logger)
    
    def visualize_column(col_name):
        if col_name not in target_df.columns:
            print_log(f"Column {col_name} not found in metadata", logger=logger)
            return
            
        labels = target_df[col_name].fillna('Unknown').values
        unique_labels = np.unique(labels)
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        int_labels = np.array([label_to_int[l] for l in labels])
        class_names = {i: l for l, i in label_to_int.items()}
        
        print_log(f"Generating {col_name} visualization with {len(unique_labels)} classes...", logger=logger)
        
        for method in ['tsne', 'umap']:
            save_path = os.path.join(save_dir, f'{bone_name}_{col_name}_{method}.png')
            create_visualization(
                target_features, 
                int_labels, 
                method=method, 
                save_path=save_path,
                class_names=class_names
            )
            print_log(f"Saved {method} plot to {save_path}", logger=logger)

    visualize_column('Order')
    visualize_column('Family')

def generate_visualizations(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    exp_dir = args.experiment_path if hasattr(args, 'experiment_path') else './experiments'
    import os
    os.makedirs(exp_dir, exist_ok=True)
    
    vis_dir = os.path.join(exp_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    feature_file = os.path.join(vis_dir, 'features.pkl')
    
    if os.path.exists(feature_file):
        print_log(f"Loading cached features from {feature_file}", logger=logger)
        try:
            with open(feature_file, 'rb') as f:
                feature_data = pickle.load(f)
            
            features = feature_data['features']
            labels = feature_data['labels']
            class_names = feature_data.get('class_names', None)
            
            print_log(f"Successfully loaded cached features with shape: {features.shape}", logger=logger)
        except Exception as e:
            print_log(f"Error loading cached features: {e}. Re-extracting features...", logger=logger)
            features = None
            labels = None
            class_names = None
    else:
        features = None
        labels = None
        class_names = None
    
    # Only extract features if not loaded from cache
    if features is None or labels is None:
        (_, train_dataloader), (_, valid_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                                builder.dataset_builder(args, config.dataset.val)
        
        print_log(f"Extracting features from dataset...", logger=logger)
        
        dataset = ConcatDataset([train_dataloader.dataset, valid_dataloader.dataset])
        dataloader = DataLoader(dataset, batch_size=config.total_bs)
        
        if class_names is None and hasattr(train_dataloader.dataset, 'get_classes'):
            class_names = train_dataloader.dataset.get_classes()
            # Reverse mapping: index -> name
            class_names = {v: k for k, v in class_names.items()}

        base_model = builder.model_builder(config.model, logger=logger)

        # load trained weights — UMAP/t-SNE should show *learned* features
        # (FIX: the checkpoint was previously never loaded here)
        if getattr(args, 'ckpts', None):
            base_model.load_model_from_ckpt(args.ckpts)
            print_log(f"Loaded checkpoint from {args.ckpts}", logger=logger)
        elif hasattr(config, 'ckpt') and config.ckpt:
            base_model.load_model_from_ckpt(config.ckpt)
            print_log(f"Loaded checkpoint from {config.ckpt}", logger=logger)
        else:
            print_log("WARNING: no --ckpts given — extracting features from an "
                      "untrained model", logger=logger)

        if torch.cuda.is_available():
            base_model = base_model.cuda()
            print_log("Model moved to CUDA", logger=logger)

        print_log("Extracting features...", logger=logger)
        features, labels = extract_features(
            base_model, dataloader, config, logger=logger
        )
        
        if features is not None and labels is not None:
            feature_data = {
                'features': features,
                'labels': labels,
                'class_names': class_names,
            }
            
            print_log(f"Saving features to {feature_file}", logger=logger)
            with open(feature_file, 'wb') as f:
                pickle.dump(feature_data, f)
    
    visualizations_created = []
    
    if features is not None and labels is not None:
        print_log(f"Features shape: {features.shape}", logger=logger)
        print_log(f"Generating visualizations...", logger=logger)
        
        umap_path = os.path.join(vis_dir, 'features_umap.png')
        if create_visualization(features, labels, method='umap', save_path=umap_path, class_names=class_names):
            visualizations_created.append(umap_path)
            print_log(f"Created UMAP visualization: {umap_path}", logger=logger)
            
        tsne_path = os.path.join(vis_dir, 'features_tsne.png')
        if create_visualization(features, labels, method='tsne', save_path=tsne_path, class_names=class_names):
            visualizations_created.append(tsne_path)
            print_log(f"Created t-SNE visualization: {tsne_path}", logger=logger)
        
        visualizations_created.append(feature_file)
    
    print_log(f"Created {len(visualizations_created)} visualization files", logger=logger)
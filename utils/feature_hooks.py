"""Feature-extraction hooks for the UMAP / t-SNE visualisations. Models call
_init_feature_extraction() in __init__ and _store_features(<global feature>) before the head.
"""

import torch


class FeatureExtractionMixin:

    def _init_feature_extraction(self):
        self.extract_features_flag = False
        self.features_for_visualization = None

    def enable_feature_extraction(self):
        self.extract_features_flag = True
        self.features_for_visualization = None

    def disable_feature_extraction(self):
        self.extract_features_flag = False
        self.features_for_visualization = None

    def get_features_for_visualization(self):
        return getattr(self, 'features_for_visualization', None)

    def _store_features(self, features):
        # getattr guard: stays safe even if _init_feature_extraction was skipped
        if getattr(self, 'extract_features_flag', False):
            if isinstance(features, torch.Tensor):
                self.features_for_visualization = features.detach()
            else:
                self.features_for_visualization = features

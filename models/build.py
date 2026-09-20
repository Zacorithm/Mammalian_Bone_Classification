from utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """Build a model from its config via the MODELS registry."""
    return MODELS.build(cfg, **kwargs)



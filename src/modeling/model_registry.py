
"""Registry of supported DraftKings prediction models."""

from src.modeling.model_adapters import (
    lightgbm_adapter,
    catboost_adapter,
)


DEFAULT_MODEL_TYPE = "catboost"

MODEL_REGISTRY = {
    "lightgbm": lightgbm_adapter,
    "catboost": catboost_adapter,
}


def get_model_adapter(model_type=DEFAULT_MODEL_TYPE):
    """Return the adapter for the requested model."""
    model_type = model_type.lower()

    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Available models: {available}"
        )

    return MODEL_REGISTRY[model_type]


def available_models():
    """Return all registered model names."""
    return sorted(MODEL_REGISTRY)

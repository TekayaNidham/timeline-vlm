"""
timeline-vlm: Temporal inference with Vision-Language Models.

Predict the year of first appearance for any image using temporal structure
discovered in VLM embedding spaces. Based on "A Matter of Time: Revealing
the Structure of Time in Vision-Language Models" (ACM MM '25).

Quick start:
    >>> from timeline_vlm import TimelinePredictor
    >>> predictor = TimelinePredictor("CLIP ViT-B/32").fit_from_precomputed("encodings")
    >>> predictor.predict("photo.jpg")
    1972
"""

__version__ = "1.0.0"

from .predictor import TimelinePredictor, predict_year


def list_models(verbose=False):
    """
    List all supported Vision-Language Models.

    Args:
        verbose: If True, include model family and architecture details.

    Returns:
        If verbose=False: list of display names (str).
        If verbose=True: dict mapping display name -> {key, family, backbone, pretrained}.
    """
    from .models.model_loader import MODEL_REGISTRY, MODEL_DISPLAY_NAMES

    if verbose:
        result = {}
        for key, (family, backbone, pretrained) in sorted(MODEL_REGISTRY.items()):
            display = MODEL_DISPLAY_NAMES.get(key, key)
            result[display] = {
                "key": key,
                "family": family,
                "backbone": backbone,
                "pretrained": pretrained,
            }
        return result

    return sorted(MODEL_DISPLAY_NAMES.values())


__all__ = [
    "TimelinePredictor",
    "predict_year",
    "list_models",
    "__version__",
]

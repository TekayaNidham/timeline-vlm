"""
Timeline-VLM: Python API for temporal inference with Vision-Language Models.

Clean, importable API for using the temporal awareness framework
in your own pipelines and research.

Usage:
    from timeline_vlm import TimelinePredictor

    # Quick prediction
    predictor = TimelinePredictor('clip-vit-b32')
    predictor.fit_from_precomputed('encodings')
    year = predictor.predict('photo.jpg')

    # Batch prediction
    years = predictor.predict_batch(['img1.jpg', 'img2.jpg'])

    # With custom settings
    predictor = TimelinePredictor(
        model='eva-clip-l14-336',
        method='bezier',
        reduce_dim=13,
        bezier_method='interpolation',
    )
    predictor.fit_from_precomputed('encodings')

    # Access the underlying timeline
    timeline_years = predictor.timeline_years
    timeline_quality = predictor.timeline_quality

    # Generate embeddings from scratch (requires model installed)
    predictor = TimelinePredictor('openclip-vit-bigg14')
    predictor.fit_from_dataset(prompt='P7')

    # Export embeddings for reuse
    predictor.save_embeddings('my_embeddings/')
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image


class TimelinePredictor:
    """
    High-level API for temporal prediction using VLMs.

    Supports three methods:
    - 'time_probing': Direct similarity matching (Eq. 1)
    - 'umap': UMAP-based 1D timeline (Section 3.3.1)
    - 'bezier': Bézier curve timeline (Section 3.3.2)
    """

    def __init__(self, model='clip-vit-b32', method='bezier',
                 device=None, prompt='P7',
                 reduce_dim=None, bezier_method='interpolation',
                 num_control_points=200):
        """
        Args:
            model: Model name (see TimelinePredictor.list_models())
            method: 'time_probing', 'umap', or 'bezier'
            device: 'cuda', 'cpu', or None (auto-detect)
            prompt: Prompt template ID (P1-P9) for time probing
            reduce_dim: KPCA dimension for Bézier. None (default) = original
                        embedding space. Set to an integer (e.g. 13) to reduce.
            bezier_method: 'interpolation' or 'nearest_neighbor'
            num_control_points: K parameter for Bézier curve
        """
        from .models.model_loader import _resolve_model_name
        self.model_name = _resolve_model_name(model)
        self.method = method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_id = prompt
        self.reduce_dim = reduce_dim if reduce_dim and reduce_dim > 0 else None
        self.bezier_method = bezier_method
        self.num_control_points = num_control_points

        # Lazy-loaded components
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._timeline = None
        self._time_embeddings = None
        self._timeline_years = None
        self._timeline_quality = None
        self._embeddings_path = None
        self._fitted = False

    def _ensure_model(self):
        """Lazy-load the VLM model."""
        if self._model is None:
            from .models.model_loader import load_model
            self._model, self._preprocess, self._tokenizer = load_model(
                self.model_name, self.device
            )

    def fit_from_precomputed(self, embeddings_path):
        """
        Fit the timeline from precomputed embeddings.

        Args:
            embeddings_path: Directory containing precomputed .npy files

        Returns:
            self (for chaining)
        """
        from .evaluation.embeddings import load_precomputed_embeddings
        self._embeddings_path = str(embeddings_path)
        data = load_precomputed_embeddings(embeddings_path, self.model_name)
        return self._fit_timeline(data['timeline_emb'], data['timeline_years'])

    def fit_from_embeddings(self, time_embeddings, years):
        """
        Fit from numpy arrays directly.

        Args:
            time_embeddings: (T, D) array
            years: List of T years

        Returns:
            self
        """
        return self._fit_timeline(time_embeddings, years)

    def fit_from_dataset(self, prompt=None, year_min=1700, year_max=2024):
        """
        Generate time embeddings from scratch and fit timeline.

        Requires the VLM model to be loadable (e.g., CLIP installed).

        Args:
            prompt: Override prompt ID (default: P7)
            year_min, year_max: Year range for time embeddings

        Returns:
            self
        """
        self._ensure_model()
        from .evaluation.embeddings import generate_time_embeddings

        pid = prompt or self.prompt_id
        from .utils.prompts import get_prompt_templates
        template = get_prompt_templates()[pid]

        years = list(range(year_min, year_max + 1))
        time_emb, years = generate_time_embeddings(
            self._model, self._tokenizer, self.model_name,
            template, years, self.device,
        )
        return self._fit_timeline(time_emb, years)

    def _fit_timeline(self, time_embeddings, years):
        """Internal: fit the chosen timeline method."""
        self._timeline_years = list(years)
        self._time_embeddings = time_embeddings

        if self.method == 'time_probing':
            self._fitted = True

        elif self.method == 'umap':
            from .evaluation.timeline_umap import UMAPTimeline
            self._timeline = UMAPTimeline()
            self._timeline_quality = self._timeline.fit(
                time_embeddings, years, model_name=self.model_name
            )
            self._fitted = True

        elif self.method == 'bezier':
            from .evaluation.timeline_bezier import BezierTimeline
            self._timeline = BezierTimeline(
                num_control_points=self.num_control_points
            )
            self._timeline_quality = self._timeline.fit(
                time_embeddings, years, reduce_dim=self.reduce_dim
            )
            self._fitted = True

        else:
            raise ValueError(
                f"Unknown method: '{self.method}'. "
                f"Choose from: 'time_probing', 'umap', 'bezier'"
            )

        return self

    def predict(self, image_or_path):
        """
        Predict the year of first appearance for a single image.

        Args:
            image_or_path: PIL Image, file path string, or numpy array

        Returns:
            int: Predicted year
        """
        results = self.predict_batch([image_or_path])
        return int(results[0])

    def predict_batch(self, images):
        """
        Predict years for multiple images.

        Args:
            images: List of PIL Images, file paths, or numpy arrays

        Returns:
            np.ndarray of predicted years
        """
        if not self._fitted:
            raise ValueError("Not fitted. Call fit_from_precomputed() or "
                             "fit_from_dataset() first.")
        self._ensure_model()

        embeddings = self._encode_images(images)
        return self._predict_from_embeddings(embeddings)

    def _predict_from_embeddings(self, embeddings):
        """Internal: predict years from precomputed image embeddings."""
        if self.method == 'time_probing':
            sims = 100.0 * (embeddings @ self._time_embeddings.T)
            indices = np.argmax(sims, axis=1)
            return np.array([self._timeline_years[i] for i in indices])

        elif self.method == 'umap':
            preds, _ = self._timeline.predict(embeddings)
            return preds

        elif self.method == 'bezier':
            if self.bezier_method == 'interpolation':
                return self._timeline.predict_interpolation(embeddings)
            else:
                return self._timeline.predict_nearest_neighbor(embeddings)

        raise ValueError(f"Unknown method: '{self.method}'")

    def predict_with_details(self, image_or_path):
        """
        Predict with full details including confidence and timing.

        Returns:
            dict with predicted_year, top_predictions (for time_probing),
            inference_ms, method, model
        """
        import time as _time
        if not self._fitted:
            raise ValueError("Not fitted.")
        self._ensure_model()

        t0 = _time.perf_counter()
        embeddings = self._encode_images([image_or_path])
        year = int(self._predict_from_embeddings(embeddings)[0])
        elapsed = (_time.perf_counter() - t0) * 1000

        result = {
            'predicted_year': year,
            'model': self.model_name,
            'method': self.method,
            'inference_ms': elapsed,
        }

        if self.method == 'time_probing':
            sims = (100.0 * (embeddings @ self._time_embeddings.T)).flatten()
            top5 = np.argsort(sims)[::-1][:5]
            result['top_predictions'] = [
                {'year': int(self._timeline_years[i]), 'score': float(sims[i])}
                for i in top5
            ]

        return result

    def _encode_images(self, images):
        """Encode images to embeddings."""
        all_emb = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            tensor = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_emb.append(emb.cpu().numpy())

        return np.vstack(all_emb)

    def evaluate(self, image_embeddings, ground_truth_years):
        """
        Evaluate on precomputed image embeddings.

        Args:
            image_embeddings: (N, D) array
            ground_truth_years: (N,) array

        Returns:
            dict with mae, tai, predictions
        """
        if not self._fitted:
            raise ValueError("Not fitted.")

        from .utils.metrics import calculate_TAI, mean_absolute_error

        gt = np.array(ground_truth_years)
        preds = self._predict_from_embeddings(image_embeddings)

        mae = mean_absolute_error(gt, preds)
        tai = float(np.mean([calculate_TAI(p, g) for p, g in zip(preds, gt)]))

        return {'mae': mae, 'tai': tai, 'predictions': preds}

    def save_embeddings(self, output_dir):
        """Save fitted timeline embeddings for reuse."""
        from .evaluation.embeddings import save_embeddings
        if self._time_embeddings is not None:
            save_embeddings(self._time_embeddings, self._timeline_years,
                            output_dir, prefix='timeline')

    @property
    def timeline_years(self):
        return self._timeline_years

    @property
    def timeline_quality(self):
        return self._timeline_quality

    @property
    def is_fitted(self):
        return self._fitted

    @staticmethod
    def list_models(verbose=False):
        """
        List all supported Vision-Language Models.

        Args:
            verbose: If True, return dict with model details.
                     If False, return sorted list of display names.

        Returns:
            list[str] or dict
        """
        from . import list_models
        return list_models(verbose=verbose)

    def __repr__(self):
        status = 'fitted' if self._fitted else 'not fitted'
        return (f"TimelinePredictor(model='{self.model_name}', "
                f"method='{self.method}', {status})")


# Convenience functions
def predict_year(image_path, model='clip-vit-b32', method='bezier',
                 embeddings_path='encodings', device=None):
    """
    One-liner to predict the year of an image.

    Args:
        image_path: Path to image file
        model: VLM model name
        method: 'time_probing', 'umap', or 'bezier'
        embeddings_path: Path to precomputed embeddings
        device: 'cuda' or 'cpu'

    Returns:
        int: Predicted year
    """
    predictor = TimelinePredictor(model=model, method=method, device=device)
    predictor.fit_from_precomputed(embeddings_path)
    return predictor.predict(image_path)

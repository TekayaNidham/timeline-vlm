"""
Time Probing: baseline temporal inference via dot-product similarity (Section 3.1).

Equation 1: y_pred = arg max_{y in Y} (I^T . T_y)

For each image, computes similarity to all year embeddings and selects
the year with highest similarity score.
"""

import os
import sys
import argparse
import time
import csv
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import load_model, is_clip_family
from utils.metrics import calculate_TAI, mean_absolute_error, print_evaluation_summary
from utils.prompts import get_prompt_templates


class TimeProbing:
    """
    Time probing evaluation for VLMs.

    Generates time embeddings for years Y_min..Y_max using a text prompt,
    then predicts the year of each image via argmax dot-product similarity.
    """

    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess, self.tokenizer = load_model(
            model_name, device
        )
        self._use_clip_tokenizer = is_clip_family(model_name)

    def encode_time_embeddings(self, years, prompt_template, batch_size=64):
        """
        Encode year prompts into text embeddings.

        Args:
            years: List of years to encode
            prompt_template: Template string with {year} placeholder
            batch_size: Batch size for encoding

        Returns:
            np.ndarray of shape (num_years, embed_dim)
        """
        all_embeddings = []

        for i in range(0, len(years), batch_size):
            batch_years = years[i:i + batch_size]
            prompts = [prompt_template.format(year=y) for y in batch_years]

            if self._use_clip_tokenizer:
                import clip
                tokens = clip.tokenize(prompts).to(self.device)
            else:
                tokens = self.tokenizer(prompts).to(self.device)

            with torch.no_grad():
                text_emb = self.model.encode_text(tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            all_embeddings.append(text_emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def predict_year(self, image_embedding, time_embeddings, years):
        """
        Predict year for an image via dot-product similarity (Eq. 1).

        Args:
            image_embedding: Image embedding (1, D)
            time_embeddings: Time embeddings (N, D)
            years: List of N candidate years

        Returns:
            predicted_year, similarity_scores
        """
        # Scaled dot product (matching CLIP's logit_scale)
        similarities = 100.0 * (image_embedding @ time_embeddings.T)
        max_idx = np.argmax(similarities)
        return years[max_idx], similarities

    def evaluate(self, dataset, prompt_template, years=None):
        """
        Evaluate time probing on full dataset.

        Args:
            dataset: TIME10kDataset or list of (path, year, ...) tuples
            prompt_template: Prompt template string
            years: Candidate years (default: 1700-2024, |Y|=325)

        Returns:
            dict with predictions, ground_truths, mae, tai, timing
        """
        if years is None:
            years = list(range(1700, 2025))  # Section 5.3: Y_min=1700, Y_max=2024

        # Encode time embeddings
        print(f"Encoding {len(years)} time embeddings...")
        t0 = time.perf_counter()
        time_embeddings = self.encode_time_embeddings(years, prompt_template)
        text_encode_time = (time.perf_counter() - t0) * 1000
        print(f"Text encoding: {text_encode_time:.0f}ms")

        # Evaluate on dataset
        predictions = []
        ground_truths = []
        classes = []
        per_image_ms = []

        for sample in tqdm(dataset, desc="Evaluating images"):
            image_path = sample[0]
            gt_year = sample[1]
            cls = sample[2] if len(sample) > 2 else 'unknown'

            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    image_emb = self.model.encode_image(image_tensor)
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    image_emb = image_emb.cpu().numpy()

                pred_year, _ = self.predict_year(image_emb, time_embeddings, years)
                elapsed = (time.perf_counter() - t0) * 1000
                per_image_ms.append(elapsed)

                predictions.append(pred_year)
                ground_truths.append(gt_year)
                classes.append(cls)

            except Exception as e:
                print(f"Warning: Failed on {image_path}: {e}")
                continue

        # Metrics
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        mae = mean_absolute_error(ground_truths, predictions)
        tai_scores = [calculate_TAI(p, g) for p, g in
                      zip(predictions, ground_truths)]

        return {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'classes': classes,
            'mae': float(mae),
            'tai': float(np.mean(tai_scores)),
            'timing': {
                'text_encode_ms': text_encode_time,
                'avg_per_image_ms': float(np.mean(per_image_ms))
                                    if per_image_ms else 0,
            },
        }

    def evaluate_from_embeddings(self, image_embeddings, image_years,
                                 time_embeddings, timeline_years):
        """
        Evaluate from precomputed embeddings (faster, no GPU needed for images).

        Args:
            image_embeddings: (N, D) image embeddings
            image_years: (N,) ground truth years
            time_embeddings: (T, D) text embeddings for candidate years
            timeline_years: (T,) candidate years

        Returns:
            dict with predictions, ground_truths, mae, tai
        """
        predictions = []
        image_years = np.array(image_years)

        t0 = time.perf_counter()
        for i in range(len(image_embeddings)):
            img_emb = image_embeddings[i:i+1]
            pred, _ = self.predict_year(img_emb, time_embeddings, timeline_years)
            predictions.append(pred)
        inference_time = (time.perf_counter() - t0) * 1000

        predictions = np.array(predictions)
        mae = mean_absolute_error(image_years, predictions)
        tai_scores = [calculate_TAI(p, g) for p, g in
                      zip(predictions, image_years)]

        return {
            'predictions': predictions,
            'ground_truths': image_years,
            'mae': float(mae),
            'tai': float(np.mean(tai_scores)),
            'timing': {
                'total_inference_ms': inference_time,
                'avg_per_image_ms': inference_time / len(image_embeddings),
            },
        }


def main():
    parser = argparse.ArgumentParser(description='Time Probing Evaluation')
    parser.add_argument('--model', type=str, default='clip-vit-b32')
    parser.add_argument('--data_path', type=str, default='data/TIME10k')
    parser.add_argument('--csv', type=str, default='data/time10k.csv')
    parser.add_argument('--prompt_id', type=str, default='P7')
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Use precomputed embeddings instead of dataset')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = get_prompt_templates()
    prompt_template = prompts[args.prompt_id]
    print(f"Model: {args.model}, Prompt: {args.prompt_id} = \"{prompt_template}\"")

    if args.embeddings_path:
        # Use precomputed embeddings
        from evaluation.embeddings import load_precomputed_embeddings
        data = load_precomputed_embeddings(args.embeddings_path, args.model)

        evaluator = TimeProbing(args.model, args.device)
        time_emb = evaluator.encode_time_embeddings(
            data['timeline_years'], prompt_template
        )
        results = evaluator.evaluate_from_embeddings(
            data['image_emb'], data['image_years'],
            time_emb, data['timeline_years']
        )
    else:
        # Load dataset and evaluate
        from data.dataset import TIME10kDataset
        csv_path = args.csv if os.path.exists(args.csv) else None
        dataset = TIME10kDataset(args.data_path, csv_path=csv_path)

        evaluator = TimeProbing(args.model, args.device)
        results = evaluator.evaluate(dataset, prompt_template)

    # Print results
    print(f"\nResults: MAE={results['mae']:.2f}, TAI={results['tai']:.3f}")
    print_evaluation_summary(
        results['predictions'], results['ground_truths'],
        f"{args.model} ({args.prompt_id})"
    )

    # Save results
    output_path = os.path.join(
        args.output_dir, f'{args.model}_{args.prompt_id}_results.csv'
    )
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Year', 'Ground_Truth', 'Error'])
        for pred, gt in zip(results['predictions'], results['ground_truths']):
            writer.writerow([pred, gt, abs(pred - gt)])
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()



import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import csv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import load_model
from data.dataset import TIME10kDataset
from utils.metrics import calculate_TAI, mean_absolute_error
from utils.prompts import get_prompt_templates


class TimeProbing:
    """Time probing evaluation for VLMs"""
    
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess, self.tokenizer = load_model(model_name, device)
        
    def encode_time_embeddings(self, years, prompt_template):
        """Encode year prompts into embeddings"""
        time_embeddings = []
        
        for year in tqdm(years, desc="Encoding time embeddings"):
            prompt = prompt_template.format(year=year)
            
            if 'clip' in self.model_name.lower():
                import clip
                text = clip.tokenize([prompt]).to(self.device)
            else:
                text = self.tokenizer([prompt]).to(self.device)
                
            with torch.no_grad():
                text_embedding = self.model.encode_text(text)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                
            time_embeddings.append(text_embedding.cpu().numpy())
            
        return np.vstack(time_embeddings)
    
    def predict_year(self, image_embedding, time_embeddings, years):
        """Predict year for an image based on similarity to time embeddings"""
        # Compute dot product similarity
        similarities = 100.0 * (image_embedding @ time_embeddings.T)
        
        # Get the year with highest similarity
        max_idx = np.argmax(similarities)
        predicted_year = years[max_idx]
        
        return predicted_year, similarities
    
    def evaluate(self, dataset, prompt_template, years=None):
        """Evaluate time probing on dataset"""
        if years is None:
            years = list(range(1700, 2025))
            
        # Encode time embeddings
        time_embeddings = self.encode_time_embeddings(years, prompt_template)
        
        # Evaluate on dataset
        predictions = []
        ground_truths = []
        
        for image_path, gt_year in tqdm(dataset, desc="Evaluating images"):
            # Encode image
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_embedding = self.model.encode_image(image_tensor)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                image_embedding = image_embedding.cpu().numpy()
            
            # Predict year
            pred_year, _ = self.predict_year(image_embedding, time_embeddings, years)
            
            predictions.append(pred_year)
            ground_truths.append(gt_year)
            
        # Calculate metrics
        mae = mean_absolute_error(ground_truths, predictions)
        tai = np.mean([calculate_TAI(pred, gt) for pred, gt in zip(predictions, ground_truths)])
        
        return {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'mae': mae,
            'tai': tai
        }


def main():
    parser = argparse.ArgumentParser(description='Time Probing Evaluation')
    parser.add_argument('--model', type=str, default='clip-vit-b32',
                        help='Model name to evaluate')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to TIME10k dataset')
    parser.add_argument('--prompt_id', type=int, default=7,
                        help='Prompt template ID (1-9)')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = TIME10kDataset(args.data_path)
    
    # Get prompt template
    prompts = get_prompt_templates()
    prompt_template = prompts[f'P{args.prompt_id}']
    
    # Initialize evaluator
    evaluator = TimeProbing(args.model, args.device)
    
    # Run evaluation
    print(f"Evaluating {args.model} with prompt: {prompt_template}")
    results = evaluator.evaluate(dataset, prompt_template)
    
    # Print results
    print(f"\nResults:")
    print(f"MAE: {results['mae']:.2f} years")
    print(f"TAI: {results['tai']:.3f}")
    
    # Save results
    output_path = os.path.join(args.output_dir, f'{args.model}_P{args.prompt_id}_results.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Predicted Year', 'Ground Truth'])
        for pred, gt in zip(results['predictions'], results['ground_truths']):
            writer.writerow(['', pred, gt])
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
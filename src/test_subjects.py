#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import time
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_specific_subjects():
    config_file = 'config/smartfallmm/student.yaml'
    model_dir = '../experiments/student_2025-05-10_07-30-58_20250510_073100/models'
    test_subjects = [38, 46]
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    sys.path.append('.')
    from models.transformer_optimized import TransModel
    from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
    from feeder.make_dataset_tf import UTD_MM_TF
    
    class ConfigObject:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    config_obj = ConfigObject(config)
    builder = prepare_smartfallmm_tf(config_obj)
    results = {}
    
    for subject in test_subjects:
        logger.info(f"\n=== Testing Subject {subject} ===")
        
        # Find appropriate model weights
        model_path = os.path.join(model_dir, f'student_model_{subject}.weights.h5')
        if not os.path.exists(model_path):
            # Use any available model
            available_models = [f for f in os.listdir(model_dir) if f.endswith('.weights.h5')]
            if available_models:
                model_path = os.path.join(model_dir, available_models[0])
                logger.warning(f"No specific model for subject {subject}, using: {model_path}")
            else:
                logger.error(f"No model weights found in {model_dir}")
                continue
        
        # Create model
        model = TransModel(**config['model_args'])
        
        # Initialize model
        dummy_input = {'accelerometer': tf.zeros((1, 64, 3))}
        _ = model(dummy_input, training=False)
        
        # Load weights
        try:
            model.load_weights(model_path)
            logger.info(f"Loaded weights from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            continue
        
        # Load data
        normalized_data = split_by_subjects_tf(builder, [subject], False)
        data_loader = UTD_MM_TF(
            dataset=normalized_data,
            batch_size=16,
            use_smv=False,  # No SMV for testing
            window_size=64
        )
        
        # Run inference
        all_predictions = []
        all_probabilities = []
        all_labels = []
        inference_times = []
        
        for batch_idx in range(len(data_loader)):
            inputs, labels, _ = data_loader[batch_idx]
            
            start_time = time.time()
            outputs = model(inputs, training=False)
            inference_time = time.time() - start_time
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            probs = tf.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int).flatten()
            
            all_predictions.extend(preds)
            all_probabilities.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())
            inference_times.append(inference_time)
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
        metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
        
        results[f'subject_{subject}'] = metrics
        
        logger.info(f"Subject {subject} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('test_subjects_results.csv')
    logger.info("\nResults saved to test_subjects_results.csv")
    
    # Display summary
    logger.info("\nFinal Results:")
    logger.info(results_df.to_string())
    
    # Average results
    avg_results = results_df.mean()
    logger.info("\nAverage Results:")
    for metric, value in avg_results.items():
        logger.info(f"  {metric}: {value:.2f}")

def calculate_metrics(labels, predictions, probabilities):
    accuracy = accuracy_score(labels, predictions) * 100
    f1 = f1_score(labels, predictions, zero_division=0) * 100
    precision = precision_score(labels, predictions, zero_division=0) * 100
    recall = recall_score(labels, predictions, zero_division=0) * 100
    
    try:
        auc = roc_auc_score(labels, probabilities) * 100
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

if __name__ == "__main__":
    test_specific_subjects()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Module for LightHART-TF

Functions for model evaluation and results reporting
"""
import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix
from trainer.training_loop import eval_step

def evaluate_test_set(model, criterion, data_loader, subject_id, work_dir, logger=None):
    """Evaluate model on test set with comprehensive reporting"""
    model.trainable = False
    
    # Initialize metrics
    test_loss = 0.0
    all_labels = []
    all_preds = []
    all_logits = []
    steps = 0
    
    # Create progress bar
    total_steps = len(data_loader)
    desc = f"Eval {name} ({epoch+1})"
    progress_bar = tqdm(data_loader, desc=desc, total=total_steps) 
    # Iterate through batches
    for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
        try:
            # Convert targets to float32
            targets = tf.cast(targets, tf.float32)
            
            # Forward pass
            loss, predictions, logits = eval_step(model, inputs, targets, criterion)
            
            # Update metrics
            loss_val = loss.numpy() if isinstance(loss, tf.Tensor) else float(loss)
            test_loss += loss_val
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            all_logits.extend(logits.numpy())
            steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{test_loss/steps:.4f}"
            })
        except tf.errors.InvalidArgumentError as e:
            if logger:
                logger(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Calculate average loss and metrics
    test_loss /= steps if steps > 0 else 1.0
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Log results
    if logger:
        logger(
            f"Test Results for Subject {subject_id}: "
            f"Loss={test_loss:.4f}, "
            f"Acc={metrics['accuracy']:.2f}%, "
            f"F1={metrics['f1']:.2f}%, "
            f"Prec={metrics['precision']:.2f}%, "
            f"Rec={metrics['recall']:.2f}%, "
            f"AUC={metrics['auc']:.2f}%"
        )
    
    # Create confusion matrix visualization
    plot_confusion_matrix(
        y_pred=all_preds,
        y_true=all_labels,
        work_dir=work_dir,
        subject_id=subject_id,
        logger=logger
    )
    
    # Save test results
    results = {
        "subject": subject_id,
        "accuracy": float(metrics['accuracy']),
        "f1": float(metrics['f1']),
        "precision": float(metrics['precision']),
        "recall": float(metrics['recall']),
        "auc": float(metrics['auc']),
        "loss": float(test_loss)
    }
    
    # Create results directory
    results_dir = os.path.join(work_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results as JSON
    results_file = os.path.join(results_dir, f'test_results_{subject_id}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Return metrics dictionary for further use
    return metrics


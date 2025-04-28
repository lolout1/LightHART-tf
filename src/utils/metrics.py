#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics Module for LightHART-TF

Contains functions for calculating and reporting metrics
"""
import numpy as np
from collections import Counter
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(targets, predictions):
    """Calculate evaluation metrics with robust handling for edge cases"""
    # Convert to numpy arrays
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
        
    # Flatten arrays
    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(targets, predictions) * 100
    
    # Handle edge cases for binary metrics
    unique_targets = np.unique(targets)
    unique_preds = np.unique(predictions)
    
    if len(unique_targets) <= 1 or len(unique_preds) <= 1:
        # Single class in targets or predictions
        if len(unique_targets) == 1 and len(unique_preds) == 1 and unique_targets[0] == unique_preds[0]:
            # Perfect prediction of a single class
            if unique_targets[0] == 1:  # All positive
                precision = 100.0
                recall = 100.0
                f1 = 100.0
            else:  # All negative
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            auc = 50.0  # Undefined AUC for single class
        else:
            # Imperfect prediction with one class
            if 1 in unique_targets:  # Positive class exists in targets
                tp = np.sum((predictions == 1) & (targets == 1))
                fn = np.sum((predictions == 0) & (targets == 1))
                fp = np.sum((predictions == 1) & (targets == 0))
                
                precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            auc = 50.0
    else:
        # Normal case with multiple classes
        try:
            precision = precision_score(targets, predictions, zero_division=0) * 100
            recall = recall_score(targets, predictions, zero_division=0) * 100
            f1 = f1_score(targets, predictions, zero_division=0) * 100
            
            try:
                auc = roc_auc_score(targets, predictions) * 100
            except:
                auc = 50.0  # Default AUC when calculation fails
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            auc = 50.0
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def add_avg_df(results):
    """Add average row to results"""
    if not results:
        return results
        
    avg_result = {'test_subject': 'Average'}
    
    for column in results[0].keys():
        if column != 'test_subject':
            values = [float(r[column]) for r in results]
            avg_result[column] = round(sum(values) / len(values), 2)
    
    results.append(avg_result)
    return results

def calculate_class_weights(labels, logger=None):
    """Calculate class weights for imbalanced training"""
    # Fix error handling for empty labels
    if labels is None or (hasattr(labels, 'size') and labels.size == 0):
        if logger:
            logger("No labels found, using default pos_weight=1.0")
        return tf.constant(1.0)

    # Count class occurrences
    label_count = Counter(labels)

    # Ensure there are positive examples
    if 1 in label_count and label_count[1] > 0 and 0 in label_count:
        # Calculate ratio of negative to positive examples
        pos_weight = float(label_count[0]) / float(label_count[1])
    else:
        pos_weight = 1.0

    # Log class distribution
    if logger:
        neg_count = label_count.get(0, 0)
        pos_count = label_count.get(1, 0)
        logger(f"Class balance - Negative: {neg_count}, Positive: {pos_count}")
        logger(f"Positive class weight: {pos_weight:.4f}")

    return tf.constant(pos_weight)

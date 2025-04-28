# src/utils/metrics.py
import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(targets, predictions, threshold=0.5):
    """Calculate metrics with robust handling for single-class data.
    
    This function properly handles the case where all samples belong to one class,
    fixing the F1 score calculation issue for subject 30.
    
    Args:
        targets: Ground truth labels
        predictions: Model predictions (can be logits or probabilities)
        threshold: Threshold for binary classification
        
    Returns:
        dict: Dictionary of metrics
    """
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()
    
    if isinstance(predictions, tf.Tensor):
        # Handle logits or probabilities
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            # Multi-class case
            pred_probs = tf.nn.softmax(predictions, axis=-1).numpy()[:, 1]
            pred_classes = (pred_probs > threshold).astype(np.int32)
        else:
            # Binary case with logits
            pred_probs = tf.sigmoid(predictions).numpy().flatten()
            pred_classes = (pred_probs > threshold).astype(np.int32)
    else:
        # Already processed predictions
        pred_classes = np.array(predictions).flatten().astype(np.int32)
        pred_probs = pred_classes.astype(np.float32)
    
    # Convert targets to flat numpy array
    targets = np.array(targets).flatten().astype(np.int32)
    
    # Calculate accuracy
    accuracy = accuracy_score(targets, pred_classes) * 100
    
    # Check for single-class targets (subject 30 issue)
    unique_targets = np.unique(targets)
    
    # Handle single-class data specially
    if len(unique_targets) == 1:
        target_class = unique_targets[0]
        logging.info(f"Single-class dataset detected (class {target_class})")
        
        if target_class == 1:  # All positive samples
            # Count true positives and false negatives
            tp = np.sum((pred_classes == 1) & (targets == 1))
            fn = np.sum((pred_classes == 0) & (targets == 1))
            total = tp + fn
            
            # Calculate precision (avoid division by zero)
            if np.sum(pred_classes == 1) == 0:
                precision = 100.0 if fn == 0 else 0.0
            else:
                precision = 100.0 * tp / np.sum(pred_classes == 1)
            
            # Calculate recall
            recall = 0.0 if total == 0 else 100.0 * tp / total
            
            # Calculate F1
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
                
            auc = 50.0  # Default AUC for single-class
        else:  # All negative samples
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            auc = 50.0
    else:
        # Standard metrics calculation
        try:
            precision = precision_score(targets, pred_classes, zero_division=0) * 100
            recall = recall_score(targets, pred_classes, zero_division=0) * 100
            f1 = f1_score(targets, pred_classes, zero_division=0) * 100
            
            if len(unique_targets) > 1:
                auc = roc_auc_score(targets, pred_probs) * 100
            else:
                auc = 50.0
        except Exception as e:
            logging.warning(f"Error calculating metrics: {e}")
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

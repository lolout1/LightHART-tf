import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(targets, predictions, threshold=0.5):
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()
    if isinstance(predictions, tf.Tensor):
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_classes = tf.argmax(predictions, axis=1).numpy()
            pred_probs = tf.nn.softmax(predictions, axis=1).numpy()[:, 1]
        else:
            pred_probs = tf.sigmoid(predictions).numpy().flatten()
            pred_classes = (pred_probs > threshold).astype(np.int64)
    else:
        pred_classes = np.array(predictions).flatten()
        pred_probs = pred_classes.astype(np.float32)
    
    targets = np.array(targets).flatten().astype(np.int64)
    
    accuracy = accuracy_score(targets, pred_classes) * 100
    
    if len(np.unique(targets)) < 2 or len(np.unique(pred_classes)) < 1:
        return {
            'accuracy': accuracy,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auc': 50.0
        }, (accuracy, 0.0, 0.0, 0.0, 50.0)
    
    f1 = f1_score(targets, pred_classes, zero_division=0) * 100
    precision = precision_score(targets, pred_classes, zero_division=0) * 100
    recall = recall_score(targets, pred_classes, zero_division=0) * 100
    
    try:
        auc = roc_auc_score(targets, pred_probs) * 100 if len(np.unique(targets)) > 1 else 50.0
    except:
        auc = 50.0
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    
    return metrics, (accuracy, f1, recall, precision, auc)

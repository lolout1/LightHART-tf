import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import os

def calculate_metrics(targets, predictions, threshold=0.5, average='binary'):
    # Convert tensors to numpy arrays
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    
    # Convert lists to numpy arrays - fixing the error
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    # Ensure arrays are flattened and have proper types (int64 for consistency)
    targets = targets.flatten().astype(np.int64)
    
    # Handle probability or class predictions
    if np.issubdtype(predictions.dtype, np.floating):
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_probs = predictions.flatten()
            pred_classes = (pred_probs > threshold).astype(np.int64)
    else:
        pred_classes = predictions.flatten().astype(np.int64)
    
    # Skip calculation if insufficient data
    if len(np.unique(targets)) < 2 or len(np.unique(pred_classes)) < 1:
        return {'accuracy': 0.0, 'f1': 0.0, 'recall': 0.0, 'precision': 0.0, 'auc': 0.5}, (0.0, 0.0, 0.0, 0.0, 0.5)
    
    # Calculate metrics
    try:
        acc = accuracy_score(targets, pred_classes)
        f1 = f1_score(targets, pred_classes, zero_division=0, average=average)
        recall = recall_score(targets, pred_classes, zero_division=0, average=average)
        precision = precision_score(targets, pred_classes, zero_division=0, average=average)
        
        # AUC calculation
        try:
            if len(np.unique(targets)) < 2:
                auc = 0.5
            else:
                if not np.issubdtype(predictions.dtype, np.floating):
                    auc = 0.5
                else:
                    auc = roc_auc_score(targets, pred_probs if 'pred_probs' in locals() else pred_classes)
        except:
            auc = 0.5
            
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'accuracy': 0.0, 'f1': 0.0, 'recall': 0.0, 'precision': 0.0, 'auc': 0.5}, (0.0, 0.0, 0.0, 0.0, 0.5)
    
    metrics = {'accuracy': acc, 'f1': f1, 'recall': recall, 'precision': precision, 'auc': auc}
    return metrics, (acc*100, f1*100, recall*100, precision*100, auc*100)

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    if labels is None:
        labels = ['Negative', 'Positive'] if cm.shape[0] == 2 else [str(i) for i in range(cm.shape[0])]
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return cm

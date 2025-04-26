from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

def calculate_metrics(targets, predictions):
    """Calculate evaluation metrics for binary classification."""
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    if len(targets) == 0 or len(predictions) == 0:
        return 0, 0, 0, 0, 0
        
    f1 = f1_score(targets, predictions, zero_division=0)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    
    try:
        auc_score = roc_auc_score(targets, predictions)
    except:
        auc_score = 0.5
        
    accuracy = accuracy_score(targets, predictions)
    
    return accuracy*100, f1*100, recall*100, precision*100, auc_score*100

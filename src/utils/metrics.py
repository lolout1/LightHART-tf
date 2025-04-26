import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

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

class BinaryFocalLossTF(tf.keras.losses.Loss):
    """Focal loss for imbalanced binary classification problems."""
    def __init__(self, alpha=0.75, gamma=2.0, from_logits=True, name='binary_focal_loss'):
        super(BinaryFocalLossTF, self).__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.from_logits:
            prob = tf.sigmoid(y_pred)
        else:
            prob = y_pred
        pt = tf.where(tf.equal(y_true, 1.0), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
        focal_loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
        return tf.reduce_mean(focal_loss)

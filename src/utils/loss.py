# src/utils/loss.py
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class BinaryFocalLoss:
    """Focal loss for binary classification"""
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        # Ensure proper shapes
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1])
        
        # Calculate probabilities
        prob = tf.nn.sigmoid(y_pred)
        
        # Calculate focal loss components
        pt = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Focal weight
        focal_weight = alpha_t * tf.pow(1 - pt, self.gamma)
        
        # Cross entropy loss
        ce_loss = -tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
        
        # Final loss
        return tf.reduce_mean(focal_weight * ce_loss)

class DistillationLoss:
    """Knowledge distillation loss with feature matching"""
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.bce = BinaryFocalLoss(alpha=0.75)
        self.epsilon = 1e-8
    
    def __call__(self, student_logits, teacher_logits, labels, 
                 teacher_features=None, student_features=None):
        """
        Calculate distillation loss
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels
            teacher_features: Feature maps from teacher (optional)
            student_features: Feature maps from student (optional)
        """
        # Ensure proper shapes
        student_logits = tf.reshape(student_logits, [-1])
        teacher_logits = tf.reshape(teacher_logits, [-1])
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)
        
        # Hard loss (student vs ground truth)
        hard_loss = self.bce(labels, student_logits)
        
        # Feature-based distillation if features available
        if teacher_features is not None and student_features is not None:
            # Calculate correct prediction mask from teacher
            teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
            correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
            
            # Weight based on teacher correctness (PyTorch: (1.0/1.5) * correct + (0.5/1.5) * incorrect)
            weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
            
            # Apply temperature scaling
            soft_teacher = tf.nn.softmax(teacher_features / self.temperature, axis=-1)
            log_soft_student = tf.nn.log_softmax(student_features / self.temperature, axis=-1)
            
            # KL divergence manually calculated (matches PyTorch version)
            kl_loss = tf.reduce_sum(
                soft_teacher * (tf.math.log(soft_teacher + self.epsilon) - log_soft_student), 
                axis=-1
            )
            
            # Average over sequence dimension
            kl_loss = tf.reduce_mean(kl_loss, axis=1)
            
            # Apply weights
            weights = tf.expand_dims(weights, -1)
            weighted_kl_loss = weights * kl_loss
            
            # Final KL loss with temperature scaling
            kl_loss = tf.reduce_mean(weighted_kl_loss) * (self.temperature ** 2)
            
            # Combine losses
            total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * hard_loss
            
            # Log loss components
            if tf.executing_eagerly():
                logger.debug(f"Hard loss: {hard_loss:.4f}, KL loss: {kl_loss:.4f}, "
                           f"Total loss: {total_loss:.4f}")
        else:
            # No feature distillation, just use hard loss
            total_loss = hard_loss
            
        return total_loss

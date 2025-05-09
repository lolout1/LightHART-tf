# utils/loss.py
import tensorflow as tf
import logging

class DistillationLoss:
    """
    Knowledge distillation loss that combines:
    1. Classification loss (BCE)
    2. Feature distillation loss (KL divergence)
    """
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        
        # Optimized for tensor operations
        logging.info(f"Distillation loss initialized with T={temperature}, alpha={alpha}, pos_weight={pos_weight}")
    
    def __call__(self, student_logits, teacher_logits, labels, teacher_features, student_features, training=True):
        """
        Calculate combined distillation loss
        
        Args:
            student_logits: Predictions from student model
            teacher_logits: Predictions from teacher model
            labels: Ground truth labels
            teacher_features: Feature representations from teacher model
            student_features: Feature representations from student model
            
        Returns:
            Combined distillation loss
        """
        # Ensure proper shapes for binary classification
        if len(tf.shape(student_logits)) > 1 and tf.shape(student_logits)[1] == 1:
            student_logits = tf.squeeze(student_logits, axis=1)
        if len(tf.shape(teacher_logits)) > 1 and tf.shape(teacher_logits)[1] == 1:
            teacher_logits = tf.squeeze(teacher_logits, axis=1)
        if len(tf.shape(labels)) > 1 and tf.shape(labels)[1] == 1:
            labels = tf.squeeze(labels, axis=1)
        
        # Binary hard target loss (BCE)
        if self.pos_weight is not None:
            hard_loss = tf.nn.weighted_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32),
                logits=student_logits,
                pos_weight=self.pos_weight
            )
        else:
            hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32),
                logits=student_logits
            )
        
        # Weight based on teacher's correct predictions
        teacher_probs = tf.sigmoid(teacher_logits)
        teacher_pred = tf.cast(teacher_probs > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, tf.cast(labels, tf.float32)), tf.float32)
        
        # Higher weight (1.0) for correct predictions, lower (0.5) for incorrect
        # Using the same weighting scheme as PyTorch implementation
        weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
        weighted_hard_loss = weights * hard_loss
        
        # Feature distillation using KL divergence
        # Flatten the features if they have more than 2 dimensions
        if len(tf.shape(teacher_features)) > 2:
            teacher_features = tf.reshape(teacher_features, 
                                         [tf.shape(teacher_features)[0], -1])
        if len(tf.shape(student_features)) > 2:
            student_features = tf.reshape(student_features, 
                                         [tf.shape(student_features)[0], -1])
        
        # Apply softmax with temperature for KL-divergence
        teacher_probs = tf.nn.softmax(teacher_features / self.temperature, axis=-1)
        student_log_probs = tf.nn.log_softmax(student_features / self.temperature, axis=-1)
        
        # Calculate KL divergence
        kl_div = tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs), 
                               axis=-1)
        
        # Apply weights to KL divergence - focusing on examples where teacher is correct
        weights_expanded = tf.expand_dims(weights, -1)
        weighted_kl_div = tf.reduce_mean(kl_div)
        
        # Scale by temperature squared as in original KD paper
        feature_loss = weighted_kl_div * (self.temperature ** 2)
        
        # Final combined loss
        total_loss = self.alpha * feature_loss + (1.0 - self.alpha) * tf.reduce_mean(weighted_hard_loss)
        
        return total_loss

class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for handling class imbalance
    This matches the PyTorch implementation but optimized for TensorFlow
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def call(self, y_true, y_pred):
        """Calculate focal loss"""
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Apply sigmoid if input is logits
        prob = tf.nn.sigmoid(y_pred)
        
        # Calculate p_t (probability of being correct class)
        p_t = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        
        # Calculate alpha_t (class balancing weight)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Calculate focal weight (down-weight easy examples)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        
        # Calculate loss
        loss = -focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        return loss

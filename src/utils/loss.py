#!/usr/bin/env python
import tensorflow as tf
import logging

logger = logging.getLogger('loss-tf')

class DistillationLoss:
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        logger.info(f"Distillation loss initialized: T={temperature}, alpha={alpha}")
    
    def __call__(self, student_logits, teacher_logits, labels, teacher_features, student_features):
        # Ensure proper shapes
        student_logits = tf.squeeze(student_logits)
        teacher_logits = tf.squeeze(teacher_logits)
        labels = tf.cast(tf.squeeze(labels), tf.float32)
        
        # Hard loss with label weighting
        hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=student_logits
        )
        
        if self.pos_weight is not None:
            hard_loss = hard_loss * (self.pos_weight * labels + (1 - labels))
        
        # Weight based on teacher's correct predictions
        teacher_probs = tf.sigmoid(teacher_logits)
        teacher_pred = tf.cast(teacher_probs > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
        
        weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
        
        # Feature distillation using KL divergence
        teacher_feat_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
        student_feat_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
        
        teacher_scaled = teacher_feat_flat / self.temperature
        student_scaled = student_feat_flat / self.temperature
        
        teacher_probs = tf.nn.softmax(teacher_scaled, axis=-1)
        student_log_probs = tf.nn.log_softmax(student_scaled, axis=-1)
        
        kl_div = tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs),
            axis=-1
        )
        
        weighted_kl = weights * kl_div * (self.temperature ** 2)
        
        total_loss = self.alpha * tf.reduce_mean(weighted_kl) + (1 - self.alpha) * tf.reduce_mean(weights * hard_loss)
        
        return total_loss

class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Apply sigmoid if input is logits
        prob = tf.nn.sigmoid(y_pred)
        
        # Calculate p_t (probability of being correct class)
        p_t = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        
        # Calculate alpha_t (class balancing weight)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        
        # Calculate loss
        loss = -focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        return loss

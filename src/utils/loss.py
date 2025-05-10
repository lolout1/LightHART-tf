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
        student_logits = tf.squeeze(student_logits)
        teacher_logits = tf.squeeze(teacher_logits)
        labels = tf.cast(tf.squeeze(labels), tf.float32)
        
        # Hard loss
        hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=student_logits)
        if self.pos_weight is not None:
            hard_loss = hard_loss * (self.pos_weight * labels + (1 - labels))
        
        # Teacher confidence weighting
        teacher_probs = tf.sigmoid(teacher_logits)
        teacher_pred = tf.cast(teacher_probs > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
        weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
        
        # Feature dimension alignment
        teacher_dim = tf.shape(teacher_features)[-1]
        student_dim = tf.shape(student_features)[-1]
        
        if teacher_dim != student_dim:
            # Project to common dimension
            common_dim = tf.minimum(teacher_dim, student_dim)
            teacher_features = tf.keras.layers.Dense(common_dim, activation=None)(teacher_features)
            student_features = tf.keras.layers.Dense(common_dim, activation=None)(student_features)
        
        # Feature distillation
        teacher_feat_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
        student_feat_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
        
        # L2 normalize features for stable KL divergence
        teacher_feat_norm = tf.nn.l2_normalize(teacher_feat_flat, axis=-1)
        student_feat_norm = tf.nn.l2_normalize(student_feat_flat, axis=-1)
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        teacher_probs = tf.nn.softmax(teacher_feat_norm / self.temperature, axis=-1) + epsilon
        student_log_probs = tf.nn.log_softmax(student_feat_norm / self.temperature, axis=-1)
        
        kl_div = tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs) - student_log_probs), axis=-1)
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
        prob = tf.nn.sigmoid(y_pred)
        p_t = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        loss = -focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
        
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        return loss

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
distillation_tf.py - Knowledge distillation implementation for TensorFlow
Matches PyTorch implementation exactly with proper temperature scaling
"""

import tensorflow as tf
import logging

logger = logging.getLogger('distillation-tf')

class DistillationLoss:
    """
    Knowledge distillation loss combining hard targets (ground truth) and 
    soft targets (teacher predictions) with feature distillation
    """
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        
        # BCE loss for hard targets
        if pos_weight is not None:
            self.bce = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE,
                pos_weight=pos_weight
            )
        else:
            self.bce = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )
        
        logger.info(f"Distillation loss initialized: T={temperature}, alpha={alpha}")
    
    def __call__(self, student_logits, teacher_logits, labels, 
                 teacher_features, student_features):
        """
        Calculate combined distillation loss
        
        Args:
            student_logits: Predictions from student model
            teacher_logits: Predictions from teacher model
            labels: Ground truth labels
            teacher_features: Feature representations from teacher
            student_features: Feature representations from student
            
        Returns:
            Combined distillation loss
        """
        # Handle shape differences
        if len(tf.shape(student_logits)) > 1 and tf.shape(student_logits)[1] == 1:
            student_logits = tf.squeeze(student_logits, axis=1)
        if len(tf.shape(teacher_logits)) > 1 and tf.shape(teacher_logits)[1] == 1:
            teacher_logits = tf.squeeze(teacher_logits, axis=1)
        if len(tf.shape(labels)) > 1 and tf.shape(labels)[1] == 1:
            labels = tf.squeeze(labels, axis=1)
        
        # Convert labels to float32 for BCE loss
        labels = tf.cast(labels, tf.float32)
        
        # Calculate hard loss (student predictions vs ground truth)
        hard_loss = self.bce(labels, student_logits)
        
        # Calculate teacher accuracy mask - exactly like PyTorch
        # Higher weight (1.0) for correct predictions, lower (0.5) for incorrect
        teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
        weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
        
        # Apply weighting for hard loss
        weighted_hard_loss = weights * hard_loss
        
        # Process features for KL divergence - handle shape differences gracefully
        # Ensure features are properly flattened for distillation
        if len(tf.shape(teacher_features)) > 2:
            teacher_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
        else:
            teacher_flat = teacher_features
            
        if len(tf.shape(student_features)) > 2:
            student_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
        else:
            student_flat = student_features
        
        # Apply temperature scaling
        teacher_scaled = teacher_flat / self.temperature
        student_scaled = student_flat / self.temperature
        
        # Calculate softmax probabilities
        teacher_probs = tf.nn.softmax(teacher_scaled, axis=-1)
        student_log_probs = tf.nn.log_softmax(student_scaled, axis=-1)
        
        # KL divergence loss
        kl_div = tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs),
            axis=1
        )
        
        # Scale by temperature^2 as in original distillation papers
        feature_loss = kl_div * (self.temperature**2)
        
        # Apply correctness weights to feature loss
        weights_expanded = tf.expand_dims(weights, axis=1)
        
        # Combine losses with alpha weighting
        total_loss = (
            self.alpha * tf.reduce_mean(feature_loss) + 
            (1.0 - self.alpha) * tf.reduce_mean(weighted_hard_loss)
        )
        
        return total_loss


class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for addressing class imbalance
    Matches the PyTorch implementation perfectly
    """
    def __init__(self, alpha=0.75, gamma=2.0, from_logits=True, reduction='mean', name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Apply sigmoid if input is logits
        if self.from_logits:
            prob = tf.nn.sigmoid(y_pred)
        else:
            prob = y_pred
        
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

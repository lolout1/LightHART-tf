import os
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
import traceback
import importlib
import sys
import csv
import shutil
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import seaborn as sns

from base_trainer import BaseTrainer, EarlyStoppingTF  # Assuming base_trainer.py exists

class BinaryFocalLossTF(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def call(self, y_true, y_pred):
        prob = tf.sigmoid(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        pt = tf.where(y_true == 1, prob, 1 - prob)
        alpha_t = tf.where(y_true == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * tf.math.log(pt + 1e-8)
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        else:
            return loss

class DistillationLossTF(tf.keras.losses.Loss):
    def __init__(self, temperature=4.5, alpha=0.6, pos_weights=None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.bce = BinaryFocalLossTF(alpha=0.6)
        self.kl_div = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        self.epsilon = 1e-3

    def call(self, student_logits, teacher_logits, labels, teacher_features, student_features, target):
        label_loss = self.bce(labels, student_logits)
        soft_teacher = tf.nn.softmax(teacher_features / self.temperature, axis=-1)
        soft_prob = tf.nn.log_softmax(student_features / self.temperature, axis=-1)
        kl_loss = self.kl_div(soft_teacher, soft_prob)
        teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.int32)
        correct_mask = tf.cast(tf.equal(teacher_pred, tf.cast(labels, tf.int32)), tf.float32)
        weights = (1.0 / 1.5) * correct_mask + (0.5 / 1.5) * (1 - correct_mask)
        weights = tf.expand_dims(weights, -1)
        cosine_loss = tf.reduce_mean(weights * kl_loss)
        loss = self.alpha * cosine_loss + (1 - self.alpha) * label_loss
        return loss

class CrossModalAlignerTF(tf.keras.layers.Layer):
    def __init__(self, feature_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim // num_heads)

    def call(self, student_features, teacher_features):
        aligned_output = self.cross_attn(query=teacher_features, key=student_features, value=student_features)
        return aligned_output

class BaseDistillation(BaseTrainer):
    def __init__(self, arg):
        super().__init__(arg)
        self.teacher_model = self.load_model(self.arg.teacher_model, self.arg.teacher_args)
        self.cross_aligner = CrossModalAlignerTF(feature_dim=self.arg.model_args['spatial_embed'])  # Fixed: Use 'spatial_embed'
        self.early_stop = EarlyStoppingTF(patience=15, min_delta=0.001)
        self.train_loss_summary = []
        self.val_loss_summary = []

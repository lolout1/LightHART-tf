#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mm_transformer.py - Teacher model for TensorFlow LightHART
Refactored reshape logic to correctly use `spatial_embed` and ensure tensor dimensions match.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import logging

logger = logging.getLogger('mm_transformer')

def get_serializable_decorator():
    if hasattr(tf.keras, 'saving') and hasattr(tf.keras.saving, 'register_keras_serializable'):
        return tf.keras.saving.register_keras_serializable
    elif hasattr(tf.keras, 'utils') and hasattr(tf.keras.utils, 'register_keras_serializable'):
        return tf.keras.utils.register_keras_serializable
    else:
        def decorator(*args, **kwargs):
            def wrap(cls):
                return cls
            logger.warning("TensorFlow version doesn't support keras serialization decorators")
            return wrap
        return decorator

register_keras_serializable = get_serializable_decorator()

@register_keras_serializable(package='Custom', name='MMTransformer')
class MMTransformer(Model):
    def __init__(
        self,
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        spatial_embed=16,
        tdepth=2,
        num_heads=2,
        mlp_ratio=2.0,
        drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.2,
        num_classes=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mocap_frames = mocap_frames
        self.acc_frames = acc_frames
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.spatial_embed = spatial_embed
        self.tdepth = tdepth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self._built_layers = False

    def build(self, input_shape):
        if not self._built_layers:
            # Class token & positional embedding
            self.temp_token = self.add_weight(
                name='temp_token',
                shape=(1, 1, self.spatial_embed),
                initializer='zeros',
                trainable=True
            )
            self.temporal_pos_embed = self.add_weight(
                name='temporal_pos_embed',
                shape=(1, 1, self.spatial_embed),
                initializer='zeros',
                trainable=True
            )
            # Spatial conv on skeleton
            self.spatial_conv = Sequential([
                layers.Conv2D(self.spatial_embed, (1, 9), padding='same', data_format='channels_first'),
                layers.BatchNormalization(axis=1), layers.ReLU(),
                layers.Conv2D(self.spatial_embed, (1, 9), padding='same', data_format='channels_first'),
                layers.BatchNormalization(axis=1), layers.ReLU()
            ], name='spatial_conv')
            # Feature encoder
            self.spatial_encoder = Sequential([
                layers.Conv1D(self.spatial_embed, 3, padding='same'),
                layers.BatchNormalization(), layers.ReLU()
            ], name='spatial_encoder')
            self.transform = layers.Dense(self.spatial_embed, activation='relu', name='transform')
            self.pos_drop = layers.Dropout(self.drop_rate)
            # Temporal blocks
            self.temporal_blocks = []
            for i in range(self.tdepth):
                dp = self.drop_path_rate * i / max(1, self.tdepth - 1)
                blk = TransformerBlock(
                    dim=self.spatial_embed,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=dp,
                    name=f'temporal_block_{i}'
                )
                self.temporal_blocks.append(blk)
            self.temporal_norm = layers.LayerNormalization(epsilon=1e-6, name='temporal_norm')
            self.class_head = Sequential([
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(self.num_classes)
            ], name='class_head')
            self._built_layers = True
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Unpack inputs
        if isinstance(inputs, dict):
            acc_data = inputs.get('accelerometer')
            skl_data = inputs.get('skeleton')
        else:
            acc_data, skl_data = inputs
        batch_size = tf.shape(skl_data)[0]
        # Process skeleton: [B, T, J, C] -> [B, C, T, J]
        x = tf.transpose(skl_data, [0, 3, 1, 2])
        x = self.spatial_conv(x, training=training)  # [B, spatial_embed, T, J]
        # Correct reshape: use spatial_embed
        x = tf.reshape(x, [batch_size, self.spatial_embed, self.mocap_frames * self.num_joints])
        x = tf.transpose(x, [0, 2, 1])  # [B, seq_len=T*J, spatial_embed]
        x = self.spatial_encoder(x, training=training)
        x = self.transform(x)
        # Class token & positional embedding
        cls = tf.repeat(self.temp_token, batch_size, axis=0)
        x = tf.concat([x, cls], axis=1)
        x = x + self.temporal_pos_embed
        x = self.pos_drop(x, training=training)
        # Temporal blocks
        for blk in self.temporal_blocks:
            x = blk(x, training=training)
        x = self.temporal_norm(x)
        seq_feats = x[:, :-1, :]
        pooled = tf.reduce_mean(seq_feats, axis=1)
        logits = self.class_head(pooled, training=training)
        if self.num_classes == 1:
            logits = tf.reshape(logits, (-1, 1))
        return logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'mocap_frames': self.mocap_frames,
            'acc_frames': self.acc_frames,
            'num_joints': self.num_joints,
            'in_chans': self.in_chans,
            'spatial_embed': self.spatial_embed,
            'tdepth': self.tdepth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate,
            'num_classes': self.num_classes
        })
        return cfg

@register_keras_serializable(package='Custom', name='TransformerBlock')
class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=attn_drop_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'),
            layers.Dropout(drop_rate),
            layers.Dense(dim),
            layers.Dropout(drop_rate)
        ])
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=False):
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        if training and self.drop_path_rate:
            y = tf.nn.dropout(y, rate=self.drop_path_rate)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y, training=training)
        if training and self.drop_path_rate:
            y = tf.nn.dropout(y, rate=self.drop_path_rate)
        return x + y


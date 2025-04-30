import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import numpy as np
import traceback
import shutil

class TransModel(tf.keras.Model):
    def __init__(
        self,
        acc_frames=64,
        num_classes=1,
        num_heads=4,
        acc_coords=3,
        embed_dim=32,
        num_layers=2,
        dropout=0.5,
        activation='relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self._build_components()
        dummy_input = tf.zeros((1, acc_frames, acc_coords), dtype=tf.float32)
        self(dummy_input, training=False)
        logging.info(f"TransModel initialized: frames={acc_frames}, coords={acc_coords}, embed_dim={embed_dim}")

    def _build_components(self):
        # Define all layers as class attributes to ensure proper tracking
        self.conv_layer = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=(8, 1),
            padding='same',
            name="conv_projection"
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        self.attention_layers = [
            layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                name=f"mha_{i}"
            ) for i in range(self.num_layers)
        ]
        self.ffn_layers = [
            tf.keras.Sequential([
                layers.Dense(self.embed_dim * 2, activation=self.activation, name=f"ffn_dense1_{i}"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.embed_dim, name=f"ffn_dense2_{i}"),
                layers.Dropout(self.dropout_rate)
            ], name=f"ffn_{i}") for i in range(self.num_layers)
        ]
        self.layer_norms = [
            [layers.LayerNormalization(epsilon=1e-6, name=f"ln{i}_{j}") for j in range(2)]
            for i in range(self.num_layers)
        ]
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        self.output_dense = layers.Dense(self.num_classes, name="output_dense")

        # Explicitly track all layers to avoid untracked resource errors
        self.tracked_components = [
            self.conv_layer,
            self.layer_norm,
            *self.attention_layers,
            *self.ffn_layers,
            *sum(self.layer_norms, []),
            self.final_norm,
            self.global_pool,
            self.output_dense
        ]

    def call(self, inputs, training=False):
        x = inputs.get('accelerometer', inputs) if isinstance(inputs, dict) else inputs
        x = tf.expand_dims(x, axis=2)
        x = self.conv_layer(x)
        x = tf.squeeze(x, axis=2)
        x = self.layer_norm(x)
        for i in range(self.num_layers):
            attn = self.attention_layers[i](x, x, training=training)
            x = self.layer_norms[i][0](x + attn)
            ffn = self.ffn_layers[i](x, training=training)
            x = self.layer_norms[i][1](x + ffn)
        x = self.final_norm(x)
        x = self.global_pool(x)
        logits = self.output_dense(x)
        return tf.reshape(logits, [-1, self.num_classes])

    def export_to_tflite(self, save_path, input_shape=(1, 64, 3)):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = save_path if save_path.endswith('.tflite') else f"{save_path}.tflite"
            logging.info(f"Exporting to TFLite: {save_path}, shape={input_shape}")

            # Define a wrapper model to ensure proper input signature and tracking
            class TFLiteModel(tf.keras.Model):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent
                    self.tracked_components = parent.tracked_components

                @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='accelerometer')])
                def call(self, inputs):
                    return self.parent({'accelerometer': inputs}, training=False)

            tflite_model = TFLiteModel(self)
            temp_dir = os.path.join(os.path.dirname(save_path), "temp_savedmodel")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

            # Save the model with signatures
            tf.saved_model.save(tflite_model, temp_dir, signatures={'serving_default': tflite_model.call})
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            tflite_content = converter.convert()

            with open(save_path, 'wb') as f:
                f.write(tflite_content)
            shutil.rmtree(temp_dir)
            logging.info(f"TFLite model successfully saved: {save_path}")
            return True
        except Exception as e:
            logging.error(f"TFLite export failed: {e}\n{traceback.format_exc()}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

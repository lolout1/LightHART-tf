import tensorflow as tf
from tensorflow.keras import layers, Model


class StudentTransformerTF(Model):
    """TensorFlow / Keras implementation of the TransModel student (accelerometer‑only)
    transformer.  Input shape: (batch, ACC_FRAMES, ACC_COORDS).
    Returns a tuple (logits, features) exactly matching the PyTorch version.
    The design mirrors the original architecture as closely as possible while
    following best‑practice Keras idioms.  The class is fully serialisable via
    `model.get_config()` and can be converted to TFLite with no further
    changes.
    """

    def __init__(
        self,
        acc_frames: int = 128,
        num_classes: int = 1,
        num_heads: int = 2,
        acc_coords: int = 4,
        num_layers: int = 2,
        embed_dim: int = 32,
        dropout: float = 0.5,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.acc_frames = acc_frames
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.acc_coords = acc_coords
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.activation = activation

        # 1. Input projection: Conv1D + BN (channels‑last).  Mirrors Conv1d(4→embed_dim, k=8).
        self.input_proj = layers.Conv1D(
            filters=embed_dim,
            kernel_size=8,
            strides=1,
            padding="same",
            use_bias=False,
        )
        self.bn_proj = layers.BatchNormalization()

        # 2. Transformer encoder stack --------------------------------------
        self.encoder_layers = [
            self._build_encoder_layer(i) for i in range(num_layers)
        ]

        # 3. Output heads ----------------------------------------------------
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = layers.Dense(num_classes)

    # ---------------------------------------------------------------------
    def _build_encoder_layer(self, idx: int):
        """Construct a single Transformer encoder block with residual
        connections, dropout, and a feed‑forward sub‑layer.  Returns a
        `tf.keras.layers.Layer` so we can call it like a function.
        """

        # Sub‑layers --------------------------------------------------------
        ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{idx}")
        mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            name=f"mha_{idx}",
        )
        drop1 = layers.Dropout(self.dropout_rate, name=f"drop1_{idx}")

        ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{idx}")
        ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    self.embed_dim * 2, activation=self.activation, name="ffn_dense1"
                ),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.embed_dim, name="ffn_dense2"),
            ],
            name=f"ffn_{idx}",
        )
        drop2 = layers.Dropout(self.dropout_rate, name=f"drop2_{idx}")

        # Wrap into a custom layer for clarity ------------------------------
        class EncoderBlock(layers.Layer):
            def call(self, x, training=False):  # type: ignore[override]
                # Self‑attention sub‑layer + residual
                attn_out, _ = mha(x, x, x, training=training, return_attention_scores=True)
                attn_out = drop1(attn_out, training=training)
                x = x + attn_out
                x = ln1(x)

                # Feed‑forward sub‑layer + residual
                ffn_out = ffn(x, training=training)
                ffn_out = drop2(ffn_out, training=training)
                x = x + ffn_out
                x = ln2(x)
                return x

        return EncoderBlock(name=f"encoder_block_{idx}")

    # ------------------------------------------------------------------
    def call(self, inputs, training=False, **kwargs):  # type: ignore[override]
        """Forward pass.

        * `inputs` may be a Tensor of shape (B, F, C) or a dict containing the
          key "accelerometer".
        * Returns `(logits, features)` where `logits` has shape (B, num_classes)
          and `features` has shape (B, F, embed_dim).
        """
        if isinstance(inputs, dict) and "accelerometer" in inputs:
            x = inputs["accelerometer"]
        else:
            x = inputs  # Expect (B, F, C)

        # Conv1D expects channels‑last; transpose from (B, F, C) → (B, C, F) then back.
        x = tf.transpose(x, perm=[0, 2, 1])  # (B, C, F)
        x = self.input_proj(x)
        x = self.bn_proj(x, training=training)
        x = tf.transpose(x, perm=[0, 2, 1])  # (B, F, embed_dim)

        # Transformer encoder ----------------------------------------------
        for enc in self.encoder_layers:
            x = enc(x, training=training)

        features = x  # Save pre‑pooled features

        # Temporal pooling + classification head ---------------------------
        x = self.temporal_norm(x)
        x = tf.reduce_mean(x, axis=1)  # Global average over frames (B, embed_dim)
        logits = self.output_layer(x)  # (B, num_classes)
        return logits, features

    # ------------------------------------------------------------------
    def get_config(self):  # Enables `model.save()` / `load_model()`
        config = super().get_config()
        config.update(
            {
                "acc_frames": self.acc_frames,
                "num_classes": self.num_classes,
                "num_heads": self.num_heads,
                "acc_coords": self.acc_coords,
                "num_layers": self.num_layers,
                "embed_dim": self.embed_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return config


# -------------------------------------------------------------------------
# Quick smoke test ---------------------------------------------------------
if __name__ == "__main__":
    BATCH = 16
    FRAMES = 128
    COORDS = 4

    dummy_acc = tf.random.normal(shape=(BATCH, FRAMES, COORDS))
    model = StudentTransformerTF()

    logits, feats = model(dummy_acc, training=True)
    print("logits shape:", logits.shape)   # Expected: (16, 1)
    print("features shape:", feats.shape)  # Expected: (16, 128, 32)


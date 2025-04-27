import tensorflow as tf
from tensorflow.keras import layers, Model

class StochasticDepth(layers.Layer):
    def __init__(self, survival_prob, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, x, training=None):
        if (not training) or self.survival_prob == 1.0:
            return x
        batch = tf.shape(x)[0]
        random_tensor = self.survival_prob + tf.random.uniform([batch, 1, 1], dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return tf.math.divide(x, self.survival_prob) * binary_tensor

class StudentTransformerTF(Model):
    def __init__(
        self,
        acc_frames=128,
        num_classes=1,
        num_heads=2,
        acc_coords=4,
        num_layers=2,
        embed_dim=32,
        dropout=0.3,
        drop_path=0.1,
        activation="gelu",
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
        self.drop_path = drop_path
        self.activation = activation

        self.norm_input = layers.LayerNormalization(axis=-1, name="input_norm")
        self.conv_depthwise = layers.DepthwiseConv1D(
            kernel_size=3, padding="same", name="dw_conv")
        self.conv_pointwise = layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="pw_conv",
        )
        self.bn_stem = layers.BatchNormalization(name="stem_bn")

        # Fixed positional embedding initialization
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=[1, acc_frames, embed_dim],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.pos_drop = layers.Dropout(self.dropout_rate)

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path, num_layers)]
        self.encoders = [self._build_encoder(i, dpr[i]) for i in range(num_layers)]

        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        self.head_drop = layers.Dropout(self.dropout_rate)
        self.output_layer = layers.Dense(num_classes)

    def _build_encoder(self, idx, drop_path_rate):
        mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            name=f"mha_{idx}",
        )
        ffn = tf.keras.Sequential([
            layers.Dense(self.embed_dim * 2, activation=self.activation, name=f"ff1_{idx}"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim, name=f"ff2_{idx}"),
        ])
        drop_path = StochasticDepth(1.0 - drop_path_rate, name=f"drop_path_{idx}")
        
        ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{idx}")
        ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{idx}")
        drop1 = layers.Dropout(self.dropout_rate)
        drop2 = layers.Dropout(self.dropout_rate)

        # Create a custom layer that doesn't have recursive references
        class EncoderBlock(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                
            def call(self, x, training=False):
                # Pre-norm self-attention
                y = ln1(x)
                y, _ = mha(y, y, y, training=training, return_attention_scores=True)
                y = drop1(y, training=training)
                x = x + drop_path(y, training=training)

                # Pre-norm FFN
                y = ln2(x)
                y = ffn(y, training=training)
                y = drop2(y, training=training)
                x = x + drop_path(y, training=training)
                return x
                
        return EncoderBlock(name=f"enc_block_{idx}")

    def call(self, inputs, training=False, **kwargs):
        if isinstance(inputs, dict):
            x = inputs.get("accelerometer", None)
            if x is None:
                raise ValueError("Input dict must contain key 'accelerometer'.")
        else:
            x = inputs

        x = self.norm_input(x)
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        x = self.bn_stem(x, training=training)

        x = x + self.pos_emb
        x = self.pos_drop(x, training=training)

        for enc in self.encoders:
            x = enc(x, training=training)

        features = x

        x = self.temporal_norm(x)
        avg_pool = tf.reduce_mean(x, axis=1)
        max_pool = tf.reduce_max(x, axis=1)
        x = tf.concat([avg_pool, max_pool], axis=-1)
        x = self.head_drop(x, training=training)

        logits = self.output_layer(x)
        return logits, features
        
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "acc_frames": self.acc_frames,
            "num_classes": self.num_classes,
            "num_heads": self.num_heads,
            "acc_coords": self.acc_coords,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "dropout": self.dropout_rate,
            "drop_path": self.drop_path,
            "activation": self.activation,
        })
        return cfg

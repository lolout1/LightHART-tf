import tensorflow as tf

class Block(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, drop=0.2, attn_drop=0.4,  drop_path=0., name=None, mlp_ratio=4., blocktype=None):
        super().__init__()

        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.attention = tf.keras.layers.MultiHeadAttention(
            key_dim=embed_dim, num_heads=num_heads, dropout=attn_drop)
        
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(
            embed_dim * mlp_ratio), drop=drop)

        self.add = tf.keras.layers.Add()

    @tf.function
    def call(self, value):
        
        norm_data = self.norm1(value)
        atn_out = self.attention(query = norm_data, value = norm_data)
        # x = atn_out + self.drop_path(atn_out)
        x = self.add([atn_out ,self.mlp(self.norm2(atn_out))])

        return x

class MLP(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_features, activation=tf.keras.activations.gelu)
        self.fc2 = tf.keras.layers.Dense(
            out_features, activation=tf.keras.activations.gelu)
        self.drop = tf.keras.layers.Dropout(drop)

    @tf.function
    def call(self, inputs, *args, **kwargs):

        x = self.fc1(inputs)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def drop_path(x, drop_prob: float = 0, training: bool = False):
    '''
    Drop path
    '''

    if drop_prob == 0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = shape = (x.shape[0],) + (1,) * (tf.size(x).numpy() - 1)
    random_tensor = keep_prob + tf.random.normal(shape=shape, dtype=x.dtype)
    random_tensor = tf.floor(random_tensor)
    output = tf.divide(x, keep_prob) * random_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    '''
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    '''

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, inputs):
        return drop_path(inputs, self.drop_prob, self.trainable)


class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs
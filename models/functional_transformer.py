import tensorflow as tf 

def encoder(x , embed_dim = 256 , num_heads = 4, drop = 0.2, attn_drop = 0.4, mlp_ratio = 4.):
    
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.MultiHeadAttention(num_heads = num_heads,key_dim =embed_dim ,dropout = attn_drop, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.02))(query = x,value = x,key = x,training = True)
    res = tf.keras.layers.Add()([x ,y])
    y= tf.keras.layers.LayerNormalization(epsilon = 1e-6)(res)
    
#    
    #mlp_layer
    
    y = tf.keras.layers.Dense(units = embed_dim*2, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.02))(res)
    y = tf.keras.layers.Dropout(rate = drop)(y)
    y = tf.keras.layers.Dense(units = embed_dim, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.02))(y)
    y = tf.keras.layers.Dropout(rate = drop)(y)
    y = tf.keras.layers.Add()([res,y])
    return y


def transformer( length = 256, acc_dim = 3, embed_dim = 256 , num_heads = 4,depth = 4, num_patch = 4,  drop = 0.2, attn_drop = 0.4, mlp_ratio = 4., num_classes = 2):
    inputs = tf.keras.Input(shape = (length, acc_dim))
    x = tf.keras.layers.Reshape((num_patch, -1))(inputs)
    x = tf.keras.layers.Dense(embed_dim)(x)

    for i in range(depth):
        x = encoder(x = x, embed_dim=embed_dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop, mlp_ratio=mlp_ratio)
    
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Flatten()(x)

    out = tf.keras.layers.Dense(num_classes)(x)
    
    return tf.keras.Model(inputs, out)
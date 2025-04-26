from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from models.modules import Block, MLP 

student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

class TransModel(tf.keras.Model):
    def __init__(self, acc_frames = 256, num_joints = 32, acc_dim = 3, num_heads = 4, mlp_ratio = 2., num_classes = 2, depth = 4,  embed_size = 256, drop = 0.2, attn_drop = 0.4, num_patch = 4):
        super().__init__()

        ## neccesaary variables
        self.num_patch = num_patch

        self.acc_embeding = tf.keras.layers.Dense(embed_size)
        
        #temporal block
        self.temporal_block = [
            Block(
                embed_dim= embed_size, num_heads=num_heads, drop = drop, attn_drop=attn_drop,  mlp_ratio=mlp_ratio
            ) for i in range(depth)
        ]

        self.acc_norm = tf.keras.layers.LayerNormalization(axis= -1)

        self.class_head = tf.keras.Sequential(layers=[
            tf.keras.layers.LayerNormalization(axis= -1),
            tf.keras.layers.Dense(num_classes)
        ])

        self.poolin_layer = tf.keras.layers.AveragePooling1D(pool_size=self.num_patch, strides=1, padding='VALID' )
        
        self.reshape = tf.keras.layers.Reshape((self.num_patch, -1))
        self.flatten = tf.keras.layers.Flatten()

        self.class_head = tf.keras.layers.Dense(num_classes)

    
    def Temporal_Model(self, x):
        b, f, c  = x.shape

        for _, blk in enumerate(self.temporal_block):

            x = blk(x)
        
        x = self.acc_norm(x)

        x = self.poolin_layer(x)

        return x
    
    @tf.function
    def call(self, inputs):
        
        # transforming into embedding'
        x = self.reshape(inputs)
        x = self.acc_embeding(x)

        #processing embedding with 
        x = self.Temporal_Model(x)
        x = self.flatten(x)

        #prediction 
        x = self.class_head(x)

        return x


if __name__ == "__main__":
      
      model = TransModel()
      acc_data = tf.random.normal([32, 128,3])
      logits = model(acc_data)

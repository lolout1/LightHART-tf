from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict
from einops import rearrange
import tensorflow as tf
from models.modules import Block, MLP, Identity, DropPath

teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="teacher",
)

class ConvTransformer(tf.keras.Model):
    def __init__(self, mocap_frames: int = 256, acc_frames: int = 256, num_joints: int = 32, in_chans: int = 3, acc_cords: int = 3, embed_size: int = 256, num_heads: int = 8, depth: int = 4, drop_rate: int = 0.2, num_patch: int = 4, attn_drop_rate: int = 0.2, drop_path_rate: int = 0.2, mlp_ratio: int = 2, num_classes: int = 11
                 ) -> None:
        '''

        Spatio-Temporal ConvTransformer

        '''
        super().__init__()

        self.embed_size = embed_size
        self.num_patch = num_patch
        self.mocap_frames = mocap_frames
        self.skl_patch_size = mocap_frames // num_patch
        self.acc_patch_size = acc_frames // num_patch
        self.skl_patch = self.skl_patch_size * (num_joints - 8 - 8)
        self.temp_frames = mocap_frames
        self.depth = depth
        self.num_joints = num_joints
        self.joint_coords = in_chans
        self.acc_frames = acc_frames
        self.acc_coords = acc_cords
        self.skl_encode_size = (self.skl_patch // (embed_size // 8))

        # norm layer
        self.spatial_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.acc_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.temp_norm = tf.keras.layers.LayerNormalization(axis = -1)

        # token
        self.temp_token = tf.Variable(tf.zeros([1, 1, embed_size], tf.float32))
        self.acc_token = tf.Variable(tf.zeros([1, 1, embed_size]), tf.float16)

        # postional embedding
        self.temporal_pos_embed = tf.Variable(
            tf.zeros([1, num_patch + 1, embed_size]))
        self.acc_pos_embed = tf.Variable(tf.zeros(1, 1, embed_size))

        # positional dropout
        self.pos_drop = tf.keras.layers.Dropout(rate=drop_rate)

        # transformation of raw skeleton and accelerometer
        self.spatial_patch_to_embedding = tf.keras.layers.Dense(embed_size)
        self.acc_coords_to_embedding = tf.keras.layers.Dense(embed_size)

        # drop rate
        adpr = [x.numpy() for x in tf.linspace(
            start=0.0, stop=drop_path_rate, num=self.depth)]

        # spatial conv
        self.spatial_conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=in_chans, kernel_size=(
                    1, 9), strides=1, padding='valid', data_format='channels_last'),
                tf.keras.layers.BatchNormalization(axis=-1),
                tf.keras.layers.Conv2D(filters=in_chans, kernel_size=(
                    1, 9), strides=1, padding='valid', data_format="channels_last"),
                tf.keras.layers.BatchNormalization(axis=-1)
            ]
        )

        # spatial encoder
        self.spatial_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(filters=self.skl_encode_size, kernel_size=1, strides=1,
                                       padding='valid', data_format="channels_last", activation='relu'),
                tf.keras.layers.BatchNormalization(axis=-1),
                tf.keras.layers.Conv1D(filters=self.skl_encode_size // 2, kernel_size=1,
                                       strides=1, padding='valid', data_format="channels_last", activation='relu'),
                tf.keras.layers.BatchNormalization(axis=-1),
                tf.keras.layers.Conv1D(filters=embed_size, kernel_size=1, strides=1,
                                       padding='valid', data_format="channels_last", activation='relu'),
                tf.keras.layers.BatchNormalization(axis=-1)
            ]
        )

        # acc encoder to change shape of data
        self.acc_encoder = tf.keras.layers.Dense(embed_size, activation='relu')

        # intertial encoder block
        self.accelerometer_blocks = [
            Block(embed_dim=embed_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i])
            for i in range(self.depth)
        ]


        #temporal encoder block
        self.temporal_block = [
            Block(embed_dim=embed_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i])
            for i in range(self.depth)
        ]

        #class head 
        self.class_head = tf.keras.Sequential(layers = [
            tf.keras.layers.LayerNormalization(axis = -1),
            tf.keras.layers.Dense(num_classes)
        ])

    def acc_model(self, x):

        b, f, e = x.shape
        class_token = tf.tile(self.acc_token, [b, 1, 1])
        x = tf.concat([x, class_token], axis=1)

        x += self.acc_pos_embed
        x = self.pos_drop(x)

        # encoder block signal
        eb_signals = []

        for _, blk in enumerate(self.accelerometer_blocks):
            x = blk(x)
            eb_signals.append(x)

        x = self.acc_norm(x)

        x = x[:, :f, :]
        x = tf.nn.avg_pool1d(
            x, x.shape[-2], strides=1, padding='VALID')
        x = tf.reshape(x, [b, e])
        return x, eb_signals
    
    def temporal_encoder(self, x, eb_signals):
        '''
        Temporal encoder with Attention Feature Fusion
        '''
        b, f, St = x.shape

        class_token = tf.tile(self.temp_token, [b, 1, 1])
        x = tf.concat([x, class_token], axis = 1)
        x += self.temporal_pos_embed
        for idx, blk in enumerate(self.temporal_block):
            acc_feature = eb_signals[idx]
            
            x = blk(x)
            x = x + acc_feature
        
        x = self.temp_norm(x)

        x = x[:, :f, :]
        x = tf.nn.avg_pool1d(
            x, x.shape[-2], strides=1, padding='VALID')
        x = tf.reshape(x, [b, St])
        return x       
        

    def call(self,inputs,training=False):

        skl_data, acc_data = inputs
        # if tensorflow lets me to pass dictionary as input
        b, f, j, c = skl_data.shape

        # exploring spatial relation
        x = self.spatial_conv(skl_data)

        # dividing into patch
        x = tf.reshape(x, [b, self.num_patch, -1])

        # matching encoder dimensions
        x = self.spatial_encoder(x)

        ### processing acc signal ###
        acc = tf.reshape(acc_data, [b, self.num_patch, -1])
        ax = self.acc_encoder(acc)
        ax, eb_signals = self.acc_model(ax)


        ## fusion stage
        x = self.temporal_encoder(x, eb_signals)
        x = x + ax
        logit = self.class_head(x)
        return logit




if __name__ == "__main__":
    model = ConvTransformer()
    skl_data = tf.random.normal([32, 128, 32, 3])
    acc_data = tf.random.normal([32, 128, 3])
    logits = model(skl_data, acc_data)

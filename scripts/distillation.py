import tensorflow as tf
import numpy as np
from tools.distiller import Distiller
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence

# models import can be variable
from models.ConvTransformer import ConvTransformer
from models.InertialTransformer import InertialTransformer
from models.functional_transformer import transformer



batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
skl_train = tf.random.normal([512, 256, 32, 3])
acc_train = tf.random.normal([512, 256, 3])
y_train = tf.random.uniform([512], minval=0, maxval=1, dtype=tf.int32)

skl_test = tf.random.normal([512, 256, 32, 3])
acc_test = tf.random.normal([512, 256, 3])
y_test = tf.random.uniform([512], minval=0, maxval=1, dtype=tf.int32)

x_train = [skl_train, acc_train]
x_test = [skl_test, acc_test]

teacher = ConvTransformer()
student = transformer()

# teacher.compile(
#     optimizer=Adam(),
#     loss=SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[SparseCategoricalAccuracy()],
# )

# teacher.fit([skl_train, acc_train], y_train, epochs = 1)
# teacher.evaluate([skl_test, acc_test], y_test)

# distiller = Distiller(student = student, teacher = teacher)

# distiller.compile(
#     optimizer = keras.optimizers.Adam(),
#     metrics = [SparseCategoricalAccuracy()],
#     student_loss_fn = SparseCategoricalCrossentropy(from_logits = True),
#     distillation_loss_fn = KLDivergence(),
#     alpha = 0.1,
#     temperature = 10
# )


# distiller.fit(x_train, y_train, epochs = 1)

# distiller.summary()
# distiller.evaluate(x_test, y_test)

student_scratch = transformer()

student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )


checkpoint_filepath = 'exp/smartfallmm/models/transformer'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    monitor = 'loss',
    mode = 'min',
    save_best_only=True,
    save_weights_only=False
)
# Train and evaluate student trained from scratch.
student_scratch.fit(acc_train, y_train, epochs=3, callbacks = [model_checkpoint_callback])
student_scratch.evaluate(acc_train, y_test)


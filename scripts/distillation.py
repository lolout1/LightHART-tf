#python package imports
from argparse import ArgumentParser
import yaml

#package imports
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence

#internal imports
from src.trainer.distiller import Distiller
from src.models.ConvTransformer import ConvTransformer
from src.models.functional_transformer import transformer
from src.utils.custom_callbacks import SaveStudentModelCallback
from src.data.processing import sf_processing
from src.utils.lite_converter import lite_converter
from src.evaluation.metrics import precision, recall, f1, print_report


# batch_size = 64
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# skl_train = tf.random.normal([512, 256, 32, 3])
# acc_train = tf.random.normal([512, 256, 3])
# y_train = tf.random.uniform([512], minval=0, maxval=1, dtype=tf.int32)

#need configs for 
# data_dir 
# model config 


def get_args():
    parser = ArgumentParser(description="Distillation")
    parser.add_argument('--config', default='./config/smartfallmm/distill.yaml')
    parser.add_argument('--teacher-args' , default= str , help = 'A dictionary for teacher args')
    parser.add_argument('--student-args' , default=str, help = ' A dictionary for student args' )
    parser.add_argument('--data-dir', default=str, help = 'root directory of datasets')
    parser.add_argument('--subjects', nargs='+', type = int)
    parser.add_argument('--processing-args' , default= str , help = 'A dictionary for data processing args')
    parser.add_argument('--epochs', default=10, help='Number of epochs')
    parser.add_argument('--teacher-path', default='exp/smartfallmm/kd/models/convtransformer')
    parser.add_argument('--student-path', default = 'exp/smartfallmm/kd/models/transformer' )
    return parser

# # Train and evaluate student trained from scratch.
# student_scratch.fit(acc_train, y_train, epochs=3, callbacks = [model_checkpoint_callback])
# student_scratch.evaluate(acc_train, y_test)

if __name__  == "__main__":
    parser = get_args()
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    train_data = sf_processing(**arg.processing_args, subjects=arg.subjects[:-2])
    test_data = sf_processing(**arg.processing_args, subjects=arg.subjects[-2:])

    x_train, y_train = [train_data['skl_data'], train_data['acc_data']], train_data['labels']
    x_test, y_test = [test_data['skl_data'], test_data['acc_data']], test_data['labels']

    teacher = ConvTransformer(**arg.teacher_args)
    # student = transformer(**arg.student_args)

    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     filepath = arg.teacher_path,
    #     monitor = 'val_loss',
    #     mode = 'min',
    #     save_best_only=True,
    #     save_weights_only=True
    #     )


    # teacher.compile(
    #     optimizer=Adam(),
    #     loss=SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[SparseCategoricalAccuracy()],
    # )

    # teacher.fit(x_train, y_train, validation_data= (x_test, y_test),  callbacks= [model_checkpoint_callback],epochs = arg.epochs)
    # teacher.load_weights(filepath=arg.teacher_path)
    # y_pred = teacher.predict(x_test)
    # print_report(y_pred, y_test)


    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     filepath = arg.student_path,
    #     monitor = 'loss',
    #     mode = 'min',
    #     save_best_only=True,
    #     save_weights_only=False
    # )

    # distiller = Distiller(student = student, teacher = teacher)

    # distiller.compile(
    #     optimizer = keras.optimizers.Adam(),
    #     metrics = [SparseCategoricalAccuracy()],
    #     student_loss_fn = SparseCategoricalCrossentropy(from_logits = True),
    #     distillation_loss_fn = KLDivergence(),
    #     alpha = 0.1,
    #     temperature = 10
    # )

    # save_model = SaveStudentModelCallback(model_to_save=distiller.student, filepath=arg.student_path)


    # distiller.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = arg.epochs, callbacks=[save_model])
    # distiller.evaluate(x_test, y_test)


    # model_test = tf.keras.models.load_model(arg.student_path, compile = True)
    # y_pred = model_test.predict(x_test[1])
    # print_report(y_pred, y_test)



    # lite_converter(file_path=arg.student_path)
    student_scratch = transformer(**arg.student_args)



    student_scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
    scartch_callback = keras.callbacks.ModelCheckpoint(
        filepath = f'{arg.student_path}_scratch',
        monitor = 'val_loss',
        mode = 'min',
        save_best_only=True,
        save_weights_only=False
        )
    student_hist = student_scratch.fit(x_train[1], y_train, validation_data=(x_test[1], y_test), callbacks=[scartch_callback],
                                       epochs=arg.epochs)
    scartch_test = tf.keras.models.load_model(f'{arg.student_path}_scratch', compile = True)
    y_pred = scartch_test.predict(x_test[1])
    print_report(y_pred, y_test)
    lite_converter(file_path=f'{arg.student_path}_scratch')

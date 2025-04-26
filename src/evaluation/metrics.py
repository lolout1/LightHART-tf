from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
import tensorflow as tf
import numpy as np

def precision(y_pred , y_true):
    y_pred = tf.argmax(tf.nn.softmax(y_pred, axis = -1), axis = 1).numpy()
    score = precision_score(y_pred=y_pred, y_true=y_true)
    return score

def recall(y_pred , y_true):
    y_pred = tf.argmax(tf.nn.softmax(y_pred, axis = -1), axis = 1).numpy()
    score = recall_score(y_pred=y_pred, y_true=y_true)
    return score

def f1(y_pred, y_true):
    y_pred = tf.argmax(tf.nn.softmax(y_pred, axis = -1), axis = 1).numpy()
    score = f1_score(y_pred=y_pred, y_true=y_true)
    return score


def print_report(y_pred , y_true):
    print(f'Precision : {precision(y_pred=y_pred, y_true=y_true)}')
    print(f'Recall : {recall(y_pred, y_true)}')
    print(f'Accuracy: {accuracy(y_pred,y_true)}')
    print(f'F1 score : {f1(y_pred, y_true)}')

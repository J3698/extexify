import tensorflow as tf
from tensorflow.keras.layers import *


def model1():
    model = tf.keras.Sequential([
            Bidirectional(LSTM(128, return_sequences = True, \
                                    input_shape=(None, 3))),
            Bidirectional(LSTM(128)),
            Dense(1098)
    ])
    return model


# Importing Libraries
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


# Exception
from src.exception import CustomException
# Logging 
from src.logger import logging



# Limiting GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])









# This script contains the attention mechanism as proposed by Bahnadau.
'''

For the project we will create a CNN Encoder to encode the images and 
an RNN Decoder to decode the image and generate captions.
This file only contains the attention mechanism.
Separate scripts are created for Encoder and Decoder
CNN Encoder : src/components/cnn_encoder.py
RNN Decoder : src/components/rnn_decoder.py

'''


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        try:
            hidden_with_time_axis = tf.expand_dims(hidden, 1)
            attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                                self.W2(hidden_with_time_axis)))
            score = self.V(attention_hidden_layer)
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * features
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return (
                context_vector,
                attention_weights
            )
        except Exception as e:
            raise CustomException(e, sys)
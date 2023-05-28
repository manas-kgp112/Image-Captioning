# Importing Libraries
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense


# Exception
from src.exception import CustomException
# Logging 
from src.logger import logging



# Limiting GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])


'''This script contains the EncoderCNN processing images using fully connected NN'''
class EncoderCNN(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(EncoderCNN, self).__init__()
        self.fc = Dense(embedding_dim)

    def call(self, x):
        logging.info("Connecting Dense Neural Nets to Feature maps.")
        try:
            x = self.fc(x)
            x = tf.nn.relu(x)
        except Exception as e:
            raise CustomException(e, sys)
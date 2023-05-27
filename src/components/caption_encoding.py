# Importing Libraries
import os
import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from dataclasses import dataclass


# Exception
from src.exception import CustomException
# Logging 
from src.logger import logging

# importing tensorflow and keras modules/API
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences



# Limiting GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])



'''Caption Encoding -> Converts captions into padded sequences for feeding into attention models'''

class CaptionEncoderConfig:
    shuffle_size = 100
    batch_size = 16


class CaptionEncoder:
    # defines the configuration of caption encoder
    def __init__(self):
        self.caption_encoder_config = CaptionEncoderConfig()

    # This function cleans the captions i.e.remove filters
    def clean_captions(self, captions_dict:dict):
        logging.info("Cleaning process of captions initiated.")
        for key, values in captions_dict.items():

            # iterating over values {captions} for single image
            for i, caption in enumerate(values):
                caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
                clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
                caption_new = ' '.join(clean_words)
                caption_new = 'startseq' + caption_new + 'endseq'
                values[i] = caption_new



    # This function is used to just train the tokenizer on our set
    def train_tokenizer(self, captions_dict:dict):
        logging.info("Tokenizer training.")

        all_captions = [caption for key, values in captions_dict.items() for caption in values]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        captions_vocab = len(tokenizer.word_index) + 1
        max_len_caption = max(len(caption.split()) for caption in all_captions)

        return (
            tokenizer,
            captions_vocab,
            max_len_caption
        )
    


    # This function tokenizes our captions and pads them to max_len
    def tokenpad_captions(self, captions_dict:dict, tokenizer, captions_vocab:int, max_len_caption:int):
        logging.info("Tokenization process initiated.")
        X, Y = list(), list()
        for keys, values in captions_dict.items():

            # iterating through captions of single image
            for caption in values:
                tokenized_caption = tokenizer.texts_to_sequences(caption)
                padded_caption = pad_sequences(tokenized_caption, maxlen=max_len_caption, padding="post")[0]
                X.append(keys)
                Y.append(padded_caption)

        
        return (
            np.array(X),
            np.array(Y)
        )
    
    # Custom function to map label and parameters
    def map_func(self, image_path, cap):
        img_tensor = np.load(image_path.decode('utf-8')+'.npy')
        return img_tensor, cap

    # This function creates dataset using the padded_tokens{Y} and image_file_paths{X}
    def create_dataset(self, X, Y):
        logging.info("Creating Tensorflow dataset.")
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.map(lambda image_path, caption_pad:tf.numpy_function(self.map_func,[image_path, caption_pad],[tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self.caption_encoder_config.shuffle_size).buffer(self.caption_encoder_config.shuffle_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset



    # This is the main function which contains all the other functions {acts as pipeline}
    def encode_captions(self, train_dict, test_dict):
        # cleaning captions
        self.clean_captions(train_dict)
        self.clean_captions(test_dict)

        # preparing tokenizer
        train_tokenizer, train_captions_vocab, train_caption_max_len = self.train_tokenizer(train_dict)
        test_tokenizer, test_captions_vocab, test_caption_max_len = self.train_tokenizer(test_dict)

        # creating token pads
        X_train, Y_train = self.tokenpad_captions(train_dict, train_tokenizer, train_captions_vocab, train_caption_max_len)
        X_test, Y_test = self.tokenpad_captions(test_dict, test_tokenizer, test_captions_vocab, test_caption_max_len)

        # dataset creation
        train_dataset = self.create_dataset(X_train, Y_train)
        test_dataset = self.create_dataset(X_test, Y_test)


        return (
            train_dataset,
            test_dataset
        )
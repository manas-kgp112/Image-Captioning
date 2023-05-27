# Importing Libraries
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm
# Importing InceptionV3, Tensorflow, Keras for image extraction
import tensorflow as tf
from tensorflow import keras
from keras.models import Model



# Exception
from src.exception import CustomException
# Logging 
from src.logger import logging



'''Image Feature Extraction -> Converts images into feature extracted tensors using pretrained InceptionV3'''

@dataclass
# ImageFeatureExtractionConfig declares the path to store the pre-processor .pkl file for image feature extraction
class ImageFeatureExtractionConfig:
    preprocessor_obj_file_path=os.path.join("data", "image_feature_extractor.pkl")
    batch_size = 16


class ImageFeatureExtraction:
    # getting configuration from ImageFeatureExtractionConfig()
    def __init__(self):
        self.image_feature_extraction_config = ImageFeatureExtractionConfig()
        



    # This function loads images into tensors and reshape them using keras API
    def convert_images_to_tensors(self, image_path:str):
        logging.info("Initiating image processing.")
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image , (299,299))
            image = keras.applications.inception_v3.preprocess_input(image)

            return (
                image, image_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

        

    # This function builds InceptionV3 model to load input and output layers for our extracted image features
    def initiate_model_building(self, image):
        logging.info("Building InceptionV3 for feature map processing.")
        try:
            model = keras.applications.InceptionV3(include_top=False, weights="imagenet")
            model_input_layer = model.input
            model_output_layer = model.layers[-1].output
            image_feature_extractor = Model(model_input_layer, model_output_layer)

            return (image_feature_extractor(image))
        except Exception as e:
            raise CustomException(e, sys)
        


    # This function extracts the feature maps using the InceptionV3 built using {initiate_model_building()}
    def extract_feature_maps(self, image_dataset, set_type:str):
        logging.info("Loading feature maps for all images.")
        try:
            for image, path in tqdm(image_dataset):
                batch_features = self.initiate_model_building(image)
                batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
                for bf, p in zip(batch_features, path):
                    path_of_feature = p.numpy().decode("utf-8")
                    save_path = f"data/FeatureMaps/{set_type}/{os.path.basename(path_of_feature)}.npy"
                    np.save(save_path, bf.numpy())
        except Exception as e:
            raise CustomException(e, sys)


        


    # This function fetches the train.csv and loads the images for tensor processing
    # This is the main function which carries the pipeline, it saves the features maps in required directories
    def initiate_image_processing(self, train_data_path, test_data_path):
        logging.info("Loading Images.")
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        # loading train/test images/captions
        train_images = train_df["Image"]
        test_images = test_df["Image"]

        # creating image_dataset
        train_image_dataset = tf.data.Dataset.from_tensor_slices(train_images)
        test_image_dataset = tf.data.Dataset.from_tensor_slices(test_images)

        # mapping tensor conversion function to TF Dataset
        train_image_dataset = train_image_dataset.map(
            self.convert_images_to_tensors,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.image_feature_extraction_config.batch_size)

        test_image_dataset = test_image_dataset.map(
            self.convert_images_to_tensors,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.image_feature_extraction_config.batch_size)

        # saving feature maps as .npy files

        self.extract_feature_maps(train_image_dataset, "train")
        self.extract_feature_maps(test_image_dataset, "test")
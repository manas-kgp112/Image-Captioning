# Importing Libraries
import os
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split


# Exception
from src.exception import CustomException
# Logging 
from src.logger import logging




'''Data Ingestion -> Collects folders containing Flickr8K images and combine them with captions.txt'''
# Data Ingestion Config declares the paths of the data available
class DataIngestionConfig:
    def __init__(self
                 ,images_path:str=os.path.join("data", "Images")
                 ,captions_path:str=os.path.join("data", "captions.txt")
                 ,train_data_path:str=os.path.join("data","Train", "train.csv")
                 ,test_data_path:str=os.path.join("data","Test", "test.csv")) -> None:
        self.images_path = images_path
        self.captions_path = captions_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

# Data Ingestion class allows to collect the data from paths specified in config
class DataIngestion:
    def __init__(self) -> None:
        # DataIngestionConfig() specifies paths for data
        self.ingestion_congig = DataIngestionConfig()   

    # initiating data collection
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion process initiated.")

        try:
            # paths for data ingestion
            images_path = self.ingestion_congig.images_path
            captions_path = self.ingestion_congig.captions_path

            # converts captions.txt into a dictionary {'img_path' : [cap1, cap2, ...]}
            image_captions = {}
            with open(captions_path, "r") as f:
                for line in f:
                    image_name, caption = line.strip().split(".jpg,")
                    image_path = os.path.join("data", "Images", f"{image_name}.jpg")
                    if image_path in image_captions:
                        image_captions[image_path].append(caption)
                    else:
                        image_captions[image_path] = [caption]
            
            # converting into train and test sets
            logging.info('Train-Test split initiated.')
            images = list(image_captions.keys())
            captions = list(image_captions.values())
            train_images, test_images, train_captions, test_captions = train_test_split(images, captions, test_size=0.2, random_state=42)
            
            train_data = dict(zip(train_images, train_captions))
            test_data = dict(zip(test_images, test_captions))
            logging.info('Data split into train-test sets.')

            # converting into dataframes and saving as .csv files
            os.makedirs(os.path.dirname(self.ingestion_congig.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_congig.test_data_path), exist_ok=True)
            train_df = pd.DataFrame(train_data.items(), columns=["Image", "Caption"])
            test_df = pd.DataFrame(test_data.items(), columns=["Image", "Caption"])

            train_df.to_csv(self.ingestion_congig.train_data_path, index=False)
            test_df.to_csv(self.ingestion_congig.test_data_path, index=False)
            logging.info(f'Train and Test .csv saved at {self.ingestion_congig.train_data_path} and {self.ingestion_congig.test_data_path}')


            return (
                self.ingestion_congig.train_data_path,
                self.ingestion_congig.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
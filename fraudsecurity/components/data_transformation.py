import sys
import os
import pickle
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from fraudsecurity.exception import CustomException
from fraudsecurity.logger import logging
from fraudsecurity.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    merchant_freq_map_path = os.path.join("artifacts", "merchant_freq_map.pkl")
    zip_freq_map_path = os.path.join("artifacts", "zip_freq_map.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def load_frequency_maps(self):
        try:
            with open(self.data_transformation_config.merchant_freq_map_path, "rb") as f:
                merchant_freq_map = pickle.load(f)

            with open(self.data_transformation_config.zip_freq_map_path, "rb") as f:
                zip_freq_map = pickle.load(f)

            return merchant_freq_map, zip_freq_map

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        Creates preprocessing pipeline using ColumnTransformer
        """
        try:
            numeric_features = [
                "amt",
                "city_pop",
                "age",
                "hour",
                "merchant_freq",
                "zip_freq"
            ]

            categorical_features = [
                "category",
                "gender",
                "is_weekend"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("onehot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info(f"Numerical columns: {numeric_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train and test datasets")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "is_fraud"

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            logging.info("Loading frequency maps")

            merchant_freq_map, zip_freq_map = self.load_frequency_maps()

            min_merchant_freq = min(merchant_freq_map.values())
            min_zip_freq = min(zip_freq_map.values())

            logging.info("Applying frequency encoding")

            # Train
            X_train["merchant_freq"] = X_train["merchant"].map(merchant_freq_map)
            X_train["zip_freq"] = X_train["zip"].map(zip_freq_map)

            # Test (handle unseen)
            X_test["merchant_freq"] = (
                X_test["merchant"].map(merchant_freq_map).fillna(min_merchant_freq)
            )
            X_test["zip_freq"] = (
                X_test["zip"].map(zip_freq_map).fillna(min_zip_freq)
            )

            # Drop original high-cardinality columns
            X_train.drop(columns=["merchant", "zip"], inplace=True)
            X_test.drop(columns=["merchant", "zip"], inplace=True)

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing on train and test data")

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

import sys
import pandas as pd
from datetime import datetime

from fraudsecurity.exception import CustomException
from fraudsecurity.logger import logging
from fraudsecurity.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")

    def predict(self, features_df: pd.DataFrame):
        try:
            logging.info("Applying preprocessing on input data")

            data_transformed = self.preprocessor.transform(features_df)

            logging.info("Making prediction")
            prediction = self.model.predict(data_transformed)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        category,
        amt,
        gender,
        city_pop,
        dob,
        hour,
        is_weekend,
        merchant,
        zip_code
    ):
        self.category = category
        self.amt = amt
        self.gender = gender
        self.city_pop = city_pop
        self.dob = dob
        self.hour = hour
        self.is_weekend = is_weekend
        self.merchant = merchant
        self.zip_code = zip_code

        # Load frequency maps
        self.merchant_freq_map = load_object("artifacts/merchant_freq_map.pkl")
        self.zip_freq_map = load_object("artifacts/zip_freq_map.pkl")

        self.min_merchant_freq = min(self.merchant_freq_map.values())
        self.min_zip_freq = min(self.zip_freq_map.values())

    def get_data_as_dataframe(self):
        try:
            dob_date = datetime.strptime(self.dob, "%Y-%m-%d")
            age_days = (datetime.today() - dob_date).days

            merchant_freq = self.merchant_freq_map.get(
                self.merchant, self.min_merchant_freq
            )
            zip_freq = self.zip_freq_map.get(
                self.zip_code, self.min_zip_freq
            )

            data = {
                "category": [self.category],
                "amt": [self.amt],
                "gender": [self.gender],
                "city_pop": [self.city_pop],
                "age": [age_days],
                "hour": [self.hour],
                "is_weekend": [self.is_weekend],
                "merchant_freq": [merchant_freq],
                "zip_freq": [zip_freq],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)

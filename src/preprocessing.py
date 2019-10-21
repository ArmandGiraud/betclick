import os

import joblib
import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

from features_utils import create_numeric, dumy_encode, encode_all, scale

SERIAL_PATH = "./serials"

class ChrunPrep():
    """This class preprocess the data with a fitting option"""
    def __init__(self):
        self.is_fitted = self.check_serials()
        
    @staticmethod
    def change_dtype(data):
        """convert data types to correct types
            we should force all columns to intended data type  in prod environment"""
        data.transaction_date = pd.to_datetime(data.transaction_date)
        data.regsitration_date = pd.to_datetime(data.regsitration_date)
        data.acquisition_channel_id = data.acquisition_channel_id.map(lambda x: str(x))
        return data

    @staticmethod
    def check_serials():
        serial_list = ["cmap", "medians", "ohe", "scaler"]
        return all([os.path.exists(os.path.join(SERIAL_PATH, s)) for s in serial_list])

    def fit(self, data):
        """pipeline process to generate the required artefacts"""
        data = self.change_dtype(data)
        data = self.drop_leaky_rows(data)

        dum = encode_all(data, fit=True)
        num = create_numeric(data, fit=True)
        feat = pd.concat([dum, num], axis=1)
        scale(feat)

        
    def fit_transform(self, data):
        """pipeline to process the data and generate serializables: 5 steps:
            - change the types
            - generate binvary variables
            - build continuous variables
            - concatenate
            - scale
            Input: pandas datafrrame as loaded by the load_data method
            Output: a 2d numpy array of feature for training"""

        data = self.change_dtype(data)
        data = self.drop_leaky_rows(data)
        
        dums = encode_all(data, fit=True)
        numeric = create_numeric(data, fit=True)
        feat = pd.concat([dums, numeric], axis=1)
        assert feat.shape[0] == dums.shape[0] == numeric.shape[0],\
            "problem in concatenation, number of rows is different"
        return scale(feat, fit=True)

    def transform(self, data):
        """pipeline to process the data with the serializables: 5 steps:
            - change the types
            - generate binvary variables
            - build continuous variables
            - concatenate
            - scale 
            Input: pandas datafrrame as loaded by the load_data method
            Output: a 2d numpy array of features for inference"""
        
        if not self.is_fitted:
            raise NotFittedError("Necessary serializable not found on disk, did you call fit?")

        data = self.change_dtype(data)
        
        dums = encode_all(data, fit=False)
        numeric = create_numeric(data, fit=False)
        feat = pd.concat([dums, numeric], axis=1)
        
        assert feat.shape[0] == dums.shape[0] == numeric.shape[0],\
        "problem in concatenation, number of rows is different"
        return scale(feat, fit=False), data.groupby("customer_key").median().index.tolist()
    
    def create_labels(self, data, churn_def:int=90):
        """create labels with the business rule of 90 days
        input a integer describing the number of inactive days"""
        data = self.drop_leaky_rows(data)
        last_date = data.sort_values("transaction_date")["transaction_date"].values[-1] # select last date
        days_since_last_activity = data.groupby("customer_key")["transaction_date"].max().map(lambda x: pd.to_datetime(last_date) - x).dt.days
        has_churned = days_since_last_activity > churn_def
        return has_churned
    
    @staticmethod
    def drop_leaky_rows(data, churn_date: str='27/06/2019'):
        """remove rows within the targeting range, to avoid having future observations in the features"""
        
        churn_date = pd.to_datetime(churn_date, infer_datetime_format=True) # 90 days before end
        leaky_rows = data.transaction_date < churn_date
        print("keeping {} non leaky transactions".format(leaky_rows.sum()))
        data = data[leaky_rows] # remove transac before 90 days to avoid target leakage
        return data

if __name__ == "__main__":
    pass
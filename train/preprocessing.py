import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

from features_utils import create_numeric, dumy_encode, encode_all, scale


class ChrunPrep():
    def __init__(self):
        self.is_fitted = False
        
    @staticmethod
    def change_dtype(data):
        data.transaction_date = pd.to_datetime(data.transaction_date)
        data.regsitration_date = pd.to_datetime(data.regsitration_date)
        data.acquisition_channel_id = data.acquisition_channel_id.map(lambda x: str(x))
        return data
        
    def fit(self, data):
        data = self.change_dtype(data)
        data = self.drop_leaky_rows(data)

        dum = encode_all(data, fit=True)
        num = create_numeric(data, fit=True)
        feat = pd.concat([dum, num], axis=1)
        scale(feat)

        self.is_fitted = True

        
    
    def transform(self, data):
        if not self.is_fitted:
            raise NotFittedError("Churn Preprocesseor was not fit...")
        data = self.change_dtype(data)
        
        dums = encode_all(data, fit=False)
        numeric = create_numeric(data, fit=False)
        feat = pd.concat([dums, numeric], axis=1)
        
        assert feat.shape[0] == dums.shape[0] == numeric.shape[0],\
        "problem in concatenation, number of rows is different"
        return scale(feat, fit=False)
        
    def fit_transform(self, data):
        self.is_fitted = True
        data = self.change_dtype(data)
        data = self.drop_leaky_rows(data)
        
        dums = encode_all(data, fit=True)
        numeric = create_numeric(data, fit=True)
        feat = pd.concat([dums, numeric], axis=1)
        assert feat.shape[0] == dums.shape[0] == numeric.shape[0],\
            "problem in concatenation, number of rows is different"
        return scale(feat, fit=True)
    
    def create_labels(self, data, churn_def:int =90):
        data = self.drop_leaky_rows(data)
        last_date = data.sort_values("transaction_date")["transaction_date"].values[-1]
        days_since_last_activity = data.groupby("customer_key")["transaction_date"].max().map(lambda x: pd.to_datetime(last_date) - x).dt.days
        has_churned = days_since_last_activity > churn_def
        return has_churned
    
    @staticmethod
    def drop_leaky_rows(data, churn_date='27/06/2019'):
        
        churn_date = pd.to_datetime(churn_date, infer_datetime_format=True) # 90 days before end
        leaky_rows = data.transaction_date < churn_date
        print("keeping {} non leaky transactions".format(leaky_rows.sum()))
        data = data[leaky_rows] # remove transac before 90 days to avoid target leakage
        return data

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    from labelling import load_data
    data = load_data("../betclic_datascience_test_churn.csv", 3000000)

    cp = ChrunPrep()
    X = cp.fit_transform(data)
    y = cp.create_labels(data)

    from modelling import fit, predict

    lr = fit(X, y)
    X_pred = load_data("../betclic_datascience_test_churn.csv", 300000)
    
    X_pred = cp.transform(X_pred)
    r = predict(X_pred, lr)

    print(r)
    #print(data_ft, data_t)
    #assert data_ft==data_t, (data_ft==data_t).sum()
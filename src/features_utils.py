
"""utilities to build features"""

import os
import datetime as dt

import joblib
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler

from download import maybe_mkdir

SERIAL_PATH = "./serials"

def dumy_encode(col, data, robust_thres: int = 100):
    """Encode a single variable:
        if values are lower than robust_thres
            the values are merged into a single column labelled ("other")
        if the other column has less than 50 occurences,
            replace the concerned value by the mode of the columns
        returns: a column with modified values
        """
    print("processing col: {}".format(col))
    other_col = col + "_other"
    save_changes = {}
    var_flat = data.groupby("customer_key")[col].max()
    var_map = (var_flat.value_counts() > robust_thres).to_dict()
    var_flat_merged = var_flat.map(lambda x: x if var_map[x] else other_col)
    v_dict = var_flat_merged.value_counts().to_dict()

    if other_col in v_dict.keys():
        save_changes = {k: other_col for k, v in var_map.items() if not v}
        print("columns {} were merged: other col has {} values".\
            format([k for k, v in var_map.items() if not v], v_dict[other_col]))
        if v_dict[other_col] < 50:
            mode = var_flat.mode().values[0]
            print("replace rare value by the mode: {}".format(mode))
            var_flat_merged[var_flat_merged == other_col] = mode
            save_changes = {k: mode for k, v in save_changes.items()}
    return var_flat_merged, {col: save_changes}

def create_numeric(data, fit: bool = True, discretize: bool = False):
    "creating numeric aggregates"
    print("creating numeric...")

    # compute the  customer age based on birth date
    data.loc[:, "age"] = (dt.datetime.now() - pd.to_datetime(data.date_of_birth, format="%Y"))\
        .dt.days/365.4
    age = data.groupby("customer_key")["age"].median().round(0)

    # compute the nb of days a customer made transactions
    transaction_nb = data.groupby("customer_key")["transaction_date"].count()

    # compute time between the first registration date and now (a proxy for customer loyalty)
    first_registration = data.groupby("customer_key")["regsitration_date"].min()
    user_time = first_registration.map(lambda x: pd.to_datetime(dt.datetime.now())- x).dt.days

    # pandas apply utils

    def get_last(data):
        "return the last element of a pandas Series"
        return data.iloc[-1]

    # compute the number of bets
    bet_nb = data.groupby("customer_key")["bet_nb"].sum()
    bet_amount = data.groupby("customer_key")["bet_amount"].apply(get_last)
    # compute the number deposits and  median amount
    deposit_nb = data.groupby("customer_key")["deposit_nb"].sum()
    deposit_amount = data.groupby("customer_key")["deposit_nb"].median()
    # _1  compute the average of Net Gaming Revenue with 
    rentability = data.groupby("customer_key")["deposit_nb"].mean()
    # compute variances, since the variation of activity is important to detect churners
    var_var = ["deposit_nb", "bet_amount", "bet_nb", "deposit_nb", "_1"]
    variances = data.groupby("customer_key")[var_var].var()
    
    def time_dif(data, last3=True):
        """utils to create time intervals between transactions
        use last_3=3 to keep the mean of the last 3 transactions only"""
        diff = list()
        for i in range(1, len(data)):
            value = (data.iloc[i] - data.iloc[i - 1]).days
            diff.append(value)
        if last3:
            return pd.Series(diff[-3:]).mean()
        return pd.Series(diff).mean()

    print("Time intervals between bets")
    days = data.sort_values("transaction_date").groupby("customer_key")["transaction_date"]\
        .apply(time_dif)
    days3 = data.sort_values("transaction_date").groupby("customer_key")["transaction_date"]\
        .apply(time_dif, last3=False)

    if fit:
        medians = variances.median().to_dict()
        medians["age"] = age.median()
        medians["days"] = days.median()
        medians["days3_med"] = days3.median()
        
        joblib.dump(medians, os.path.join(SERIAL_PATH, "medians"))
    else:
        medians = joblib.load(os.path.join(SERIAL_PATH, "medians"))

    # replace missing values generated by feature engineering and write them to disk
    age = age.fillna(medians["age"])
    days = days.fillna(medians["days"])
    days3 = days3.fillna(medians["days3_med"])

    variances = variances.apply(lambda x: x.fillna(medians[x.name]), axis=0)
    variances.columns = [m + "_var" for m in variances.columns]
    assert variances.isna().sum().sum() == 0, "some missing values remains in the dataset"

    res = pd.DataFrame({
        "age":age,
        "transaction_nb":transaction_nb,
        "bet_nb":bet_nb,
        "bet_amount": bet_amount,
        "deposit_nb": deposit_nb,
        "deposit_amount": deposit_amount,
        "user_time": user_time,
        "rentability": rentability,
        "days_interval": days,
        "days3": days3
    })

    res = pd.merge(res, variances, left_index=True, right_index=True).reset_index(drop=True)
    


    if discretize:
        # utility to create buckets out of continuous features
        # seems not very useful as a whole, might be on case by case

        raise NotImplementedError("discretize not giving better results...")
        print("discretizing...") 
        est = KBinsDiscretizer(n_bins=12,
                               strategy='quantile',
                               encode="onehot-dense")
        return pd.DataFrame(est.fit_transform(res))

    return res

def encode_all(data, fit=True):
    """encode categorical variables into one hot 
     and store medians to serialziabless"""

    to_encode = ["gender", "acquisition_channel_id", "betclic_customer_segmentation"]

    if fit:
        processed_cat = [dumy_encode(d, data) for d in to_encode]
        cols, cmap = list(zip(*processed_cat))
        
        res = pd.concat(cols, axis=1)
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        dums = pd.DataFrame(ohe.fit_transform(res))
        
        # serialize data
        maybe_mkdir(SERIAL_PATH)
        joblib.dump(ohe, os.path.join(SERIAL_PATH, "ohe"))
        # unpack list of dict to pandas replace format
        cmap = {k: v for d in cmap for k, v in d.items()} 
        joblib.dump(cmap, os.path.join(SERIAL_PATH, "cmap"))
        return dums
    ohe = joblib.load(os.path.join(SERIAL_PATH, "ohe"))
    cmap = joblib.load(os.path.join(SERIAL_PATH, "cmap"))
    flat = data.replace(cmap).groupby("customer_key")[to_encode].max()
    return pd.DataFrame(ohe.transform(flat))


def scale(data, fit=True):
    """Scale all variables between 0 and 1 by using the max as upper bound
    using scikitkearn utils"""
    
    print("scaling...")
    if fit:
        scaler = MinMaxScaler()
        res = scaler.fit_transform(data)
        joblib.dump(scaler, os.path.join(SERIAL_PATH, "scaler"))
    else:
        scaler = joblib.load(os.path.join(SERIAL_PATH, "scaler"))
        res = scaler.transform(data)
    return res

if __name__ == "__main__":
    pass


import joblib
import pandas as pd
import datetime as dt

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

def dumy_encode(col, data, robust_thres:int = 100):
    print("encoding col: {}".format(col))
    other_col = col + "_other"
    save_changes = {}
    var_flat = data.groupby("customer_key")[col].max()
    var_map = (var_flat.value_counts() > robust_thres).to_dict()
    var_flat_merged = var_flat.map(lambda x: x if var_map[x] else other_col)
    v_dict = var_flat_merged.value_counts().to_dict()

    if other_col in v_dict.keys():
        save_changes = {k: other_col for k, v in var_map.items() if not v}
        print("columns {} were merged: other col has {} values".format([k for k, v in var_map.items() if not v], v_dict[other_col]))
        if v_dict[other_col] < 50:
            mode = var_flat.mode().values[0]
            print("replace rare value by the mode: {}".format(mode))
            var_flat_merged[var_flat_merged==other_col] = mode
            save_changes = {k: mode for k, v in save_changes.items()}

    return var_flat_merged, {col: save_changes}

def create_numeric(data, fit: bool=True, discretize: bool=False):
    "creating numeric aggregates"
    print("creating numeric...")
    
    now = dt.datetime.now()
    age = (now - pd.to_datetime(data.date_of_birth, format="%Y")).dt.days/365.4

    data.loc[:, "age"] = age
    age = data.groupby("customer_key")["age"].median().round(0)

    transaction_nb = data.groupby("customer_key")["transaction_date"].count()

    first_registration = data.groupby("customer_key")["regsitration_date"].min()
    user_time = first_registration.map(lambda x: pd.to_datetime(now)- x).dt.days

    def get_last(data):
        return data.iloc[-1]
    
    # bets
    bet_nb = data.groupby("customer_key")["bet_nb"].sum()
    bet_amount = data.groupby("customer_key")["bet_amount"].apply(get_last)
    
    # deposits
    deposit_nb = data.groupby("customer_key")["deposit_nb"].sum()
    deposit_amount = data.groupby("customer_key")["deposit_nb"].median()
    
    # variances
    deposit_var = data.groupby("customer_key")["deposit_nb"].var()
    bet_var = data.groupby("customer_key")["bet_amount"].var()
    bet_nb_var = data.groupby("customer_key")["bet_nb"].var()
    bet_amount_var = data.groupby("customer_key")["deposit_nb"].var()
    
    if fit:
        
        medians = dict(
            age_med = age.median(),
            dep_med = deposit_var.median(),
            betnb_med = bet_nb_var.median(),
            betvar_med = bet_var.median(),
            betam_var = bet_amount_var.median()
        )
        
        joblib.dump(medians, "medians")
    else:
        medians = joblib.load("medians")
        
    age = age.fillna(medians["age_med"])
    deposit_var = deposit_var.fillna(medians["dep_med"])
    bet_nb_var = bet_nb_var.fillna(medians["betnb_med"])
    bet_var = bet_var.fillna(medians["betvar_med"])
    bet_amount_var = bet_amount_var.fillna(medians["betam_var"])
        
    res = pd.DataFrame({
        "age":age,
        "transaction_nb":transaction_nb,
        "bet_nb":bet_nb,
        "bet_amount": bet_amount,
        "deposit_nb": deposit_nb,
        "deposit_amount": deposit_amount,
        "user_time": user_time,
        "bet_var": bet_var,
        "bet_nb_var": bet_nb_var,
        "deposit_var": deposit_var,
        "bet_amount_var":bet_amount_var
    })
    
    
    if discretize:
        raise NotImplementedError("discretize not giving better results...")
        print("discretizing...")
        est = KBinsDiscretizer(n_bins=12,
                               strategy='quantile',
                               encode="onehot-dense")
        return pd.DataFrame(est.fit_transform(res))

    return res.reset_index(drop=True)

def encode_all(data, fit=True):
    to_encode = ["gender","acquisition_channel_id", "betclic_customer_segmentation"]

    if fit:
        r = [dumy_encode(d, data) for d in to_encode]
        cols, cmap = list(zip(*r))
        
        res = pd.concat(cols, axis=1)
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        dums = pd.DataFrame(ohe.fit_transform(res))
        
        # serialize data
        joblib.dump(ohe, "ohe")
        cmap = {k: v for d in cmap for k, v in d.items()} # unpack list of dict to pandas replace format
        joblib.dump(cmap, "cmap")
        return dums
    else:
        print("loading artefacts...")
        ohe = joblib.load("ohe")
        cmap = joblib.load("cmap")
        flat = data.replace(cmap).groupby("customer_key")[to_encode].max()
        return pd.DataFrame(ohe.transform(flat))

if __name__ == "__main__":
    pass
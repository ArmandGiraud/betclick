import pandas as pd
import datetime as dt

from sklearn.preprocessing import KBinsDiscretizer

# basic features

def encode(data, col: str, robust_thres: int=100):
    "encode to dummy if enough examples"
    print("encoding col: {}".format(col))
    
    var_flat = data.groupby("customer_key")[col].max()
    var_map = (var_flat.value_counts() > robust_thres).to_dict()
    var_flat_merged = var_flat.map(lambda x: x if var_map[x] else "other")
    v_dict = var_flat_merged.value_counts().to_dict()
    
    if 'other' in v_dict.keys():
        print("columns {} were merged: other col has {} values".format([k for k, v in var_map.items() if not v], v_dict["other"]))
        if v_dict["other"] < 50:
            mode = var_flat.mode()
            print("replace rare value by the mode: {}".format(mode))
            var_flat_merged[var_flat_merged=="other"] = mode
    
    return pd.get_dummies(var_flat_merged)

def pipe_data(data):
    to_encode = ["gender", "acquisition_channel_id", "betclic_customer_segmentation"]
    encoded = [encode(data, col) for col in to_encode]
    dums = pd.concat(encoded, axis=1)

    print("data is now shaped: {}".format(dums.shape))
    return dums.reset_index(drop=True)

def create_numeric(data, discretize: bool=True):
    "creating numeric aggregates"
    print("creating numeric...")
    now = dt.datetime.now()
    age = (now - pd.to_datetime(data.date_of_birth, format="%Y")).dt.days/365.4
    age = age.fillna(age.median()) # replace missing values

    data["age"] = age
    age = data.groupby("customer_key")["age"].median().round(0)

    transaction_nb = data.groupby("customer_key")["transaction_date"].count()

    first_registration = data.groupby("customer_key")["regsitration_date"].min()
    user_time = first_registration.map(lambda x: pd.to_datetime(now)- x).dt.days

    def get_last(data):
        return data.iloc[-1]

    bet_nb = data.groupby("customer_key")["bet_nb"].sum()
    bet_amount = data.groupby("customer_key")["bet_amount"].apply(get_last)

    deposit_nb = data.groupby("customer_key")["deposit_nb"].sum()
    deposit_amount = data.groupby("customer_key")["deposit_nb"].median()
    
    res = pd.DataFrame({
        "age":age,
        "transaction_nb":transaction_nb,
        "bet_nb":bet_nb,
        "bet_amount": bet_amount,
        "deposit_nb": deposit_nb,
        "deposit_amount": deposit_amount,
        "user_time": user_time
    })
    print([r.shape for r in res])
    if discretize:
        print("discretizing...")
        est = KBinsDiscretizer(n_bins=12,
                               strategy='quantile',
                               encode="onehot-dense")
        return pd.DataFrame(est.fit_transform(res))
    return res


if __name__ == "__main__":
    from labelling import add_labels, load_data
    data = load_data("../betclic_datascience_test_churn.csv", 300000)
    data = add_labels(data)
    dums, numeric = pipe_data(data), create_numeric(data, discretize=False)
    feat = pd.concat([dums, numeric], axis=1)

    y = data.groupby("customer_key")["target"].max()
    print(y.shape, feat.shape)
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e-2)

    print(cross_val_score(lr, feat, y, cv=5, n_jobs=-1, scoring="f1").mean())
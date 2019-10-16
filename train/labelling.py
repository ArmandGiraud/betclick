import pandas as pd

CHURN_DEF=90
SAMP_SIZE = 50000

def load_data(train_path: str, samp_size: int):
    print("loading...")
    bc = pd.read_csv(train_path, sep=";").sample(samp_size)
    return bc

def add_labels(data):
    "add labels as target and drop leaky rows"
    data.transaction_date = pd.to_datetime(data.transaction_date)
    data.regsitration_date = pd.to_datetime(data.regsitration_date)
    print("labelling...")
    last_date = data.sort_values("transaction_date")["transaction_date"].values[-1]
    days_since_last_activity = data.groupby("customer_key")["transaction_date"].max().map(lambda x: pd.to_datetime(last_date) - x).dt.days
    has_churned = days_since_last_activity > CHURN_DEF

    labels = has_churned.to_dict()
    data.loc[:, "target"] = data.customer_key.map(labels)
    churn_date = pd.to_datetime('27/06/2019', infer_datetime_format=True) # 90 days before end
    leaky_rows = data.transaction_date < churn_date
    print("keeping {} non leaky transactions".format(leaky_rows.sum()))
    data = data[leaky_rows] # remove transac before 90 days to avoid target leakage
    return data


if __name__ == "__main__":
    data = load_data("../betclic_datascience_test_churn.csv", SAMP_SIZE)
    data = add_labels(data)
    print(data.head())
import pandas as pd


def load_data(train_path: str, samp_size: int):
    print("loading...")
    bc = pd.read_csv(train_path, sep=";").sample(samp_size)
    return bc

if __name__ == "__main__":
    data = load_data("../betclic_datascience_test_churn.csv", 5000)
    print(data.head())
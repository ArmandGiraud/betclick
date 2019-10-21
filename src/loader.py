import pandas as pd


def load_data(train_path: str, samp_size: int=0, all_: bool=True):

    print("loading...")
    bc = pd.read_csv(train_path, sep=";")
    
    if all_:
        return bc
    return bc.sample(samp_size)

if __name__ == "__main__":
    data = load_data("../betclic_datascience_test_churn.csv", 5000)
    print(data.head())
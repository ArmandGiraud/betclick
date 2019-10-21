"""Utility to load data"""
import pandas as pd


def load_data(train_path: str, samp_size: int = 0, all_: bool = True):
    """utils to load data with pandas"""
    print("loading...")
    betclick = pd.read_csv(train_path, sep=";")

    if all_:
        return betclick
    return betclick.sample(samp_size)

if __name__ == "__main__":
    pass

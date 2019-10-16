import pandas as pd
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--train_file",
                    type=str,
                    default="betclic_datascience_test_churn.csv",
                    help="dataset path")

args = parser.parse_args()




### from

if __name__ == "__main__":
    train()
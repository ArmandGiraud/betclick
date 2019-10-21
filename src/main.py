"""Main application file"""

import argparse
import warnings
import os

import pandas as pd
#internal import

from download import download_extract
from loader import load_data
from preprocessing import ChrunPrep
from modelling import Modelling


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_path",
                    type=str,
                    default="./data",
                    help="dataset path")

parser.add_argument("-m", "--model_type",
                    type=str,
                    default="etf",
                    help="""type of model to fit (one of [lr, gb, etf])
                            lr: LogisticRegressuion
                            gb: Gradient Boosted Trees
                            etf: ExtraTreeClassifier""")

parser.add_argument("-p", "--predict",
                    action="store_true",
                    help="whether to fit the model on data or predict a sample")

parser.add_argument("-o", "--out_path",
                    type=str,
                    default="./preds",
                    help="folder to write predictions")

parser.add_argument("-f", "--private_file",
                    type=str,
                    default=False,
                    help="folder to write predictions")

args = parser.parse_args()

DATA_NAME = os.path.join(args.data_path, "betclic_datascience_test_churn.csv")


def fit(args):
    "fit preprocessor and model"
    data = load_data(DATA_NAME, samp_size=10000, all_=False)

    prep = ChrunPrep()
    X = prep.fit_transform(data)
    y = prep.create_labels(data)

    classifier = Modelling(model=args.model_type)
    classifier.fit(X, y)

def predict(args):
    "perdict on a given dataset"
    if not args.private_file:
        data = load_data(DATA_NAME, samp_size=100000, all_=False)
    else:
        data = load_data(args.private_file)

    prep = ChrunPrep()
    X, index = prep.transform(data)

    classifier = Modelling(args.model_type)
    preds = classifier.predict(X)
    out_path = os.path.join(args.out_path, "preds.csv")
    pd.Series(preds, index=index).to_csv(out_path, sep=";")



def main(args):
    """main function define the whole pipeline"""
    download_extract(args.data_path)
    if args.private_file and (not args.predict):
        userinput = input("are you sure to fit your data on {} [Y/n]"\
            .format(args.private_file))
        if userinput.lower() != "y":
            print("stopping..")
            return 1

    if args.predict:
        predict(args)
    else:
        fit(args)
    return 0


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main(args)

import pandas as pd
import argparse
import warnings
import os

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

parser.add_argument("-p", "--private_file",
                    type=str,
                    default=False,
                    help="folder to write predictions")

args = parser.parse_args()

dir_path = args.data_path
save_path = os.path.join(dir_path, "betclick.zip")
data_path = os.path.join(dir_path, "betclic_datascience_test_churn.csv")
out_path = os.path.join(args.out_path, "preds.csv")
    
cp = ChrunPrep()


def fit(args):
    data = load_data(data_path, samp_size=10000, all_=False)

    
    X = cp.fit_transform(data)
    y = cp.create_labels(data)

    mo = Modelling(model= args.model_type)
    mo.fit(X, y)


def predict(args):
    if not args.private_file:
        data = load_data(data_path, samp_size=100000, all_=False)
    else:
        data = load_data(args.private_file)
        
    cp = ChrunPrep()
    X, index = cp.transform(data)

    mo = Modelling(args.model_type)
    preds = mo.predict(X)
    pd.Series(preds, index=index).to_csv(out_path, sep=";")



def main(args):
    if args.predict:
        predict(args)
    else:
        fit(args)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    download_extract(args.data_path)
    main(args)

    
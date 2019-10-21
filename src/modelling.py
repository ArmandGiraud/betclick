"""training and inference module"""
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

SERIAL_PATH = "./serials"

class Modelling():
    def __init__(self, model: str = "lr"):
        self.model = model

    def fit(self, X, y):
        """fit the data on chosen classfier"""
        self.init_classifier()
        self.clf.fit(X, y)
        print("serializing model...")
        joblib.dump(self.clf, os.path.join(SERIAL_PATH, "clf"))

    def predict(self, X):
        """perdict with loaded classifier"""
        self.clf = joblib.load(os.path.join(SERIAL_PATH, "clf"))

        preds = self.clf.predict_proba(X)
        is_churner = preds[:, 1] > 0.46
        return is_churner

    
    def init_classifier(self):
        clf_map = {
            "lr": LogisticRegression(C=1e10,
                                     solver="liblinear",
                                     class_weight="balanced",
                                     max_iter=1e5),

            "gb": GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           learning_rate=0.026324967240336495, loss='deviance',
                           max_depth=None, max_features=0.5135820722988753,
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=35,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=78, n_iter_no_change=None,
                           presort='auto', random_state=1,
                           subsample=0.7542915097577992, tol=0.0001,
                           validation_fraction=0.1, verbose=0),

            "etf": ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                     max_depth=None, max_features=0.9159744542919692,
                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                     min_impurity_split=None, min_samples_leaf=4,
                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                     n_estimators=23, n_jobs=-1, oob_score=False, random_state=0,
                     verbose=False, warm_start=False)
        }
        if self.model not in ["lr", "gb", "etf"]:
            raise ValueError('Wrong value for model type,\
                 it should be in one of ["lr", "gb", "etf"]')
        
        self.clf = clf_map[self.model]

    def check_fitted(self):
        return os.path.exists(os.path.join(SERIAL_PATH, "clf"))



if __name__ == "__main__":
    pass
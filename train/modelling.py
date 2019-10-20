from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def fit(X, y):
    lr = LogisticRegression(C=10)
    lr.fit(X, y)
    return lr

def predict(X, lr):
    return lr.predict(X)
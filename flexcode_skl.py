from sklearn.base import BaseEstimator
from flexcode import FlexCodeModel
from flexcode.regression_models import NN

class FlexCodeSKL(BaseEstimator):
    def __init__(self, k=20):
        self.k = k

    def fit(self, x_train, y_train):
        self.fcmodel = FlexCodeModel(NN,
            max_basis=30, basis_system="Fourier",
            regression_params={"k": self.k}, z_min=0,
            z_max=1)

        n_train = round(x_train.shape[0] * 0.9)
        x_val, y_val = x_train[:n_train], y_train[:n_train]
        x_train, y_train = x_train[n_train:], y_train[n_train:]

        self.fcmodel.fit(x_train, y_train)
        self.fcmodel.tune(x_val, y_val)

        return self

    def predict(self, x_pred):
        return self.fcmodel.predict(x_pred, n_grid=1000)

    def score(self, x_test, y_test):
        return - self.fcmodel.estimate_error(x_test, y_test)

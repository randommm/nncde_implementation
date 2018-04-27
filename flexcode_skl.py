from sklearn.base import BaseEstimator
from flexcode import FlexCodeModel
from flexcode.regression_models import NN, RandomForest, XGBoost
import numpy as np

class SKLFlexCodeBase(BaseEstimator):
    def fit(self, x_train, y_train):
        self._create_model()

        n_val = round(min(x_train.shape[0] * 0.10, 5000))
        n_train = x_train.shape[0] - n_val

        x_val, y_val = x_train[n_train:], y_train[n_train:]
        x_train, y_train = x_train[:n_train], y_train[:n_train]

        self.fcmodel.fit(x_train, y_train)
        self.fcmodel.tune(x_val, y_val)

        return self

    def predict(self, x_pred):
        #prediction = np.empty((x_pred.shape[0], 1000))
        #step = 10
        #print("O")
        #for i in range(0, x_pred.shape[0], step):
            #self.fcmodel.predict(x_pred[i:i+step], n_grid=1000)
            ##prediction[i:i+step] = pred[0].copy()
            #if i == 0:
                #pred = self.fcmodel.predict(x_pred[i:i+step], n_grid=1000)
                #grid = pred[1].copy()
                #del(pred)

        #return prediction, grid
        return self.fcmodel.predict(x_pred, n_grid=1000)

    #def score2(self, x_test, y_test):
    #    prediction, grid = self.fcmodel.predict(x_test, n_grid=1000)
    #    score1 = 23
    #    return - self.fcmodel.estimate_error(x_test, y_test)

    def score(self, x_test, y_test):
        return - self.fcmodel.estimate_error(x_test, y_test, n_grid = 1000)

class SKLFlexCodeKNN(SKLFlexCodeBase):
    def __init__(self, k=5, max_basis=30):
        self.max_basis = max_basis
        self.k = k

    def _create_model(self):
        self.fcmodel = FlexCodeModel(NN,
                    max_basis=self.max_basis, basis_system="Fourier",
                    regression_params={"k": self.k},
                    z_min=0, z_max=1)

class SKLFlexCodeXGBoost(SKLFlexCodeBase):
    def __init__(self, max_depth=6, eta=0.3, silent=1,
                 objective='reg:linear', max_basis=30):
        self.max_basis = max_basis
        self.max_depth = max_depth
        self.eta = eta
        self.silent = silent
        self.objective = objective

    def _create_model(self):
        self.fcmodel = FlexCodeModel(XGBoost,
                          max_basis=self.max_basis,
                          basis_system="Fourier",
                          regression_params = {
                             'max_depth' : self.max_depth,
                             'eta' : self.eta,
                             'silent' : self.silent,
                             'objective' : self.objective,
                          },
                          z_min=0, z_max=1)

class SKLFlexCodeRandomForest(SKLFlexCodeBase):
    def __init__(self, n_estimators=10, max_basis=30):
        self.max_basis = max_basis
        self.n_estimators = n_estimators

    def _create_model(self):
        self.fcmodel = FlexCodeModel(RandomForest,
                    max_basis=self.max_basis, basis_system="Fourier",
                    regression_params = {
                       'n_estimators' : self.n_estimators,
                    },
                    z_min=0, z_max=1)

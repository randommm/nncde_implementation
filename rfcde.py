from flexcode.loss_functions import cde_loss
from sklearn.base import BaseEstimator
import statsmodels.api as sm
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np

class RFCDE(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        importr("RFCDE")
        ro.r('y_train <- rep(0.0, ' + str(y_train.shape[0]) + ')')
        np.asarray(ro.r['y_train'])[:] = y_train[:,0]

        ro.r('x_train <- matrix(0.0, ' + str(x_train.shape[0]) + ', '+
              str(x_train.shape[1]) + ')')
        np.asarray(ro.r['x_train'])[:] = x_train

        ro.reval("""
        n_trees <- 1000
        mtry <- 4
        node_size <- 20
        n_basis <- 15

        forest <- RFCDE(x_train, y_train, n_trees = n_trees,
                        mtry = mtry, node_size = node_size,
                        n_basis = n_basis)
        """)

        self.grid = np.linspace(0, 1, 1000)[1:-1]
        ro.r('grid <- rep(0.0, ' + str(self.grid.shape[0]) + ')')
        np.asarray(ro.r['grid'])[:] = self.grid

        return self

    def predict(self, x_pred):
        ro.r('x_pred <- matrix(0.0, ' + str(x_pred.shape[0]) + ', '+
              str(x_pred.shape[1]) + ')')
        np.asarray(ro.r['x_pred'])[:] = x_pred
        importr("RFCDE")
        ro.r("""
        bandwidth <- 0.4
        density <- predict(forest, x_pred, grid, bandwidth = bandwidth)
        """)

        density = np.array(ro.r['density'])
        density /= density.sum(1)[:,None]

        return density

    def score(self, x_test, y_test):
        return cde_loss(self.predict(x_test), self.grid, y_test)

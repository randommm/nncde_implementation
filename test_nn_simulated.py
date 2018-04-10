#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import torch
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats

from nncde import NNCDE, NNCDECached, set_cache_dir
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import hashlib
import pickle
from sklearn.externals import joblib
import os

set_cache_dir("nncde_fs_cache", bytes_limit=30*2**30)

from generate_data import generate_data, true_pdf_calc

n_train = 100_000
n_test = 1_000
x_train, y_train = generate_data(n_train)
x_test, y_test = generate_data(n_test)

print(y_train)
print(min(y_train))
print(max(y_train))

ncomponents = 20

nnf_cv_obj = NNCDECached(
ncomponents=ncomponents,
verbose=2,
beta_loss_penal_exp=0.0,
beta_loss_penal_base=0.0,
nn_weights_loss_penal=0.0,
es=True,
hls_multiplier=2,
nhlayers=5,
#gpu=False,
)

cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

gs_params = dict(
  ncomponents = np.array([5, 10, 15, 20]),
  nhlayers = [4, 2, 0],
  hls_multiplier = [2],
)

name = "nn"
h = hashlib.new('ripemd160')
h.update(pickle.dumps(x_train))
h.update(pickle.dumps(y_train))
h.update(pickle.dumps(gs_params))
filename = ("nncde_fs_cache/model_" + name + "_" +
            h.hexdigest() + ".pkl")
if not os.path.isfile(filename):
    print("Started working on file", filename)
    nnf_cv_obj = GridSearchCV(nnf_cv_obj, gs_params, cv=cv, pre_dispatch=1, verbose=100)
    nnf_cv_obj.fit(x_train, y_train)

    joblib.dump(nnf_cv_obj, filename)
    print("Saved file", filename)
else:
    nnf_cv_obj = joblib.load(filename)
    print("Loaded file", filename)

nnf_obj = nnf_cv_obj.best_estimator_

#Check without using true density information
print("Score (utility) on train:", nnf_obj.score(x_train, y_train))
print("Score (utility) on test:", nnf_obj.score(x_test, y_test))

#Check using true density information
est_pdf = nnf_obj.predict(x_test)[:, 1:-1]
true_pdf = true_pdf_calc(x_test, nnf_obj.y_grid[1:-1][:,None]).T
sq_errors = (est_pdf - true_pdf)**2
#print("Squared density errors for test:\n", sq_errors)
print("\nAverage squared density errors for test:\n", sq_errors.mean())

import matplotlib.pyplot as plt
plt.plot(true_pdf[1])
plt.plot(est_pdf[1])
plt.show()

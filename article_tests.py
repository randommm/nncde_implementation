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

from nnflexcode import NNFlexCode, set_cache_dir
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import hashlib
import pickle
from sklearn.externals import joblib
import os

set_cache_dir("nnflexcode_fs_cache", bytes_limit=30*2**30)

from generate_data import generate_data, true_pdf_calc

n_train = 100_000
n_test = 800
x_train, y_train = generate_data(n_train)
x_test, y_test = generate_data(n_test)

print(y_train)
print(min(y_train))
print(max(y_train))

ncomponents = 5

nnf_obj = NNFlexCode(
ncomponents=ncomponents,
verbose=2,
beta_loss_penal_exp=0.0,
beta_loss_penal_base=0.0,
nn_weights_loss_penal=0.0,
es=True,
hls_multiplier=5,
nhlayers=10,
gpu=False,
)

nnf_obj.fit(x_train, y_train)

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

"""
cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

gs_params_list = [
  dict(
  ncomponents = np.arange(400, 20, -10),
  ),
  dict(
  nn_weights_loss_penal = np.concatenate([np.logspace(-10, 0.1, 30),
                                          np.logspace(0, 1, 30)]),
  ),
  dict(
  beta_loss_penal_exp = np.concatenate([np.logspace(-10, 0.1, 7),
                                          np.logspace(0, 1, 7)]),
  beta_loss_penal_base = np.concatenate([np.logspace(-10, 0.1, 7),
                                          np.logspace(0, 1, 7)]),
  ),
]

for gs_params in gs_params_list:
    h = hashlib.new('ripemd160')
    h.update(pickle.dumps(x_train))
    h.update(pickle.dumps(y_train))
    h.update(pickle.dumps(gs_params))

    filename = ("nnflexcode_fs_cache/model_" + h.hexdigest() + ".pkl")
    if not os.path.isfile(filename):
        print("Started working on file", filename)
        gs_clf = GridSearchCV(nnf_obj, gs_params,
                              cv=cv, verbose=100)
        gs_clf.fit(x_train, y_train)

        joblib.dump(gs_clf, filename)
        print("Saved file", filename)
    else:
        gs_clf = joblib.load(filename)
        print("Loaded file", filename)
"""

"""
gs_params = dict(
ncomponents = np.arange(500, 10, -10),
beta_loss_penal_exp = stats.truncnorm(a=-1.1, b=np.inf, scale=0.5,
                                      loc=1.1),
beta_loss_penal_base = stats.truncnorm(a=0, b=np.inf, scale=0.5),
nn_weights_loss_penal = stats.truncnorm(a=0, b=np.inf, scale=2.0),
nhlayers = stats.nbinom(n=5.0, p=1/3), #mean 10, variance 30
)

h = hashlib.new('ripemd160')
h.update(pickle.dumps(x_train))
h.update(pickle.dumps(y_train))
i = 0
gs_clf_list = []
cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for i in range(10):
    filename = ("nnflexcode_fs_cache/model_" + h.hexdigest() + "_"
                + str(i) + ".pkl")
    if not os.path.isfile(filename):
        print("Started working on file", filename)
        gs_clf = RandomizedSearchCV(nnf_obj, gs_params, n_iter=10,
                                    cv=cv, verbose=100)
        gs_clf.fit(x_train, y_train)

        joblib.dump(gs_clf, filename)
        print("Saved file", filename)
    else:
        gs_clf = joblib.load(filename)
        print("Loaded file", filename)
    gs_clf_list.append(gs_clf)
    i += 1

gs_clf.predict([x_test, y_test])
gs_clf.score(x_test, y_test)
"""

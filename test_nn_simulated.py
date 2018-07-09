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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hashlib
import pickle
from sklearn.externals import joblib
import os

set_cache_dir("nncde_fs_cache", bytes_limit=30*2**30)

from generate_data import generate_data, true_pdf_calc

n_train = 10_000
n_test = 5_000
x_train, y_train = generate_data(n_train)
x_test, y_test = generate_data(n_test)

print(y_train)
print(min(y_train))
print(max(y_train))

ncomponents = 100

nnf_obj = NNCDE(
ncomponents=ncomponents,
verbose=2,
beta_loss_penal_exp=0.0,
beta_loss_penal_base=0.0,
nn_weight_decay=0.0,
es=True,
hls_multiplier=25,
nhlayers=10,
#gpu=False,
)

nnf_obj = Pipeline([('stand', StandardScaler()),
                    ('nnf_obj', nnf_obj)])

nnf_obj.fit(x_train, y_train)

#Check without using true density information
print("Score (utility) on train:", nnf_obj.score(x_train, y_train))
print("Score (utility) on test:", nnf_obj.score(x_test, y_test))

#Check using true density information
est_pdf = nnf_obj.predict(x_test)[:, 1:-1]
true_pdf = true_pdf_calc(x_test, nnf_obj.steps[1][1].y_grid[1:-1][:,None]).T
sq_errors = (est_pdf - true_pdf)**2
#print("Squared density errors for test:\n", sq_errors)
print("\nAverage squared density errors for test:\n", sq_errors.mean())

import matplotlib.pyplot as plt
plt.plot(true_pdf[1])
plt.plot(est_pdf[1])
plt.show()

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

from nncde import NNCDE, set_cache_dir
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import hashlib
import pickle
from sklearn.externals import joblib
import os
import pandas as pd

set_cache_dir("nncde_fs_cache", bytes_limit=30*2**30)

np.random.seed(10)
df = pd.read_csv("spectroscopic.txt", ' ')
ndf = np.random.permutation(df)
y_train = np.array(ndf)[:100000,-1:]
x_train = np.array(ndf)[:100000,:-1]

n_test = round(min(x_train.shape[0] * 0.10, 5000))
n_train = x_train.shape[0] - n_test
x_test, y_test = x_train[n_train:], y_train[n_train:]
x_train, y_train = x_train[:n_train], y_train[:n_train]

print(y_train)
print(min(y_train))
print(max(y_train))

ncomponents = 30

nnf_obj = NNCDE(
ncomponents=ncomponents,
verbose=2,
beta_loss_penal_exp=0.0,
beta_loss_penal_base=0.0,
nn_weight_decay=0.0,
es=True,
hls_multiplier=50,
nhlayers=10,
#gpu=False,
)

nnf_obj.fit(x_train, y_train)

#Check without using true density information
print("Score (utility) on train:", nnf_obj.score(x_train, y_train))
print("Score (utility) on test:", nnf_obj.score(x_test, y_test))

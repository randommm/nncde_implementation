#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 2 or 3 the License.
#
#Obs.: note that the other files are licensed under GNU GPL 3. This
#file is licensed under GNU GPL 2 or 3 for compatibility with flexcode
#license only.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You can get a copy of the GNU General Public License version 2 at
#<http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import torch
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import hashlib
import pickle
from sklearn.externals import joblib
import os
import pandas as pd

from rfcde import RFCDE

np.random.seed(10)
df = pd.read_csv("dbs/spectroscopic.csv", ' ')
ndf = np.random.permutation(df)
y_train = np.array(ndf)[:10000,-1:]
x_train = np.array(ndf)[:10000,:-1]

n_test = round(min(x_train.shape[0] * 0.10, 5000))
n_train = x_train.shape[0] - n_test
x_test, y_test = x_train[n_train:], y_train[n_train:]
x_train, y_train = x_train[:n_train], y_train[:n_train]

fcs_obj = RFCDE()

fcs_obj.fit(x_train, y_train)

print("Score (utility) on test:", fcs_obj.score(x_test, y_test))


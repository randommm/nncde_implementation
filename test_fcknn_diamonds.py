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

from flexcode_skl import SKLFlexCodeKNN

np.random.seed(10)
df = pd.read_csv("dbs/diamonds.csv")
for column in ['cut', 'color', 'clarity']:
    new_df = pd.get_dummies(df[column], dummy_na=True,
                            drop_first=True, prefix=column)
    df = pd.concat([df, new_df], axis=1)
    df = df.drop(column, 1)

ndf = df.reindex(np.random.permutation(df.index))
y_train = np.array(ndf[["price"]])
x_train = np.array(ndf.drop("price", 1).iloc[:, 1:])

y_train = np.log(y_train + 0.001)
y_train_min = np.min(y_train)
y_train_max = np.max(y_train)
y_train = (y_train - y_train_min) / (y_train_max - y_train_min)
y_train = (y_train + 0.01) / 1.0202

n_test = round(min(x_train.shape[0] * 0.10, 5000))
n_train = x_train.shape[0] - n_test
x_test, y_test = x_train[n_train:], y_train[n_train:]
x_train, y_train = x_train[:n_train], y_train[:n_train]

fcs_cv_obj = SKLFlexCodeKNN()

cv = ShuffleSplit(n_splits=1, test_size=n_test, random_state=0)

gs_params = dict(
  k = np.array([1, 5, 15, 25, 35, 45, 55, 65, 75]),
)

name = "fcknn"
h = hashlib.new('ripemd160')
h.update(pickle.dumps(x_train))
h.update(pickle.dumps(y_train))
h.update(pickle.dumps(gs_params))
filename = ("nncde_fs_cache/fcs_cv_obj_" + name + "_" +
            h.hexdigest() + ".pkl")
if not os.path.isfile(filename):
    print("Started working on file", filename)
    fcs_cv_obj = GridSearchCV(fcs_cv_obj, gs_params, cv=cv, n_jobs=1,
                           verbose=100)
    fcs_cv_obj.fit(x_train, y_train)

    joblib.dump(fcs_cv_obj, filename)
    print("Saved file", filename)
else:
    fcs_cv_obj = joblib.load(filename)
    print("Loaded file", filename)

fcs_obj = fcs_cv_obj.best_estimator_

print("Score (utility) on test:", fcs_obj.score(x_test, y_test))


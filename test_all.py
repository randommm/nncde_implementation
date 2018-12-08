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

from nncde import NNCDE, set_cache_dir
from flexcode_skl import SKLFlexCodeRandomForest
from flexcode_skl import SKLFlexCodeKNN
from flexcode_skl import SKLFlexCodeXGBoost
from prepare_pnad import get_pnad_db

from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import numpy as np
from scipy import stats
import hashlib
import pickle
import os
import time
import pandas as pd

def call_test(dataset, model):
    np.random.seed(10)
    if dataset == "diamonds":
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
    elif dataset == "pnad":
        df = get_pnad_db()
        ndf = np.random.permutation(df)
        y_train = np.array(ndf)[:,-1:]
        x_train = np.array(ndf)[:,:-1]

        y_train = np.log(y_train + 0.001)
        y_train_min = np.min(y_train)
        y_train_max = np.max(y_train)
        y_train = (y_train - y_train_min) / (y_train_max - y_train_min)
        y_train = (y_train + 0.01) / 1.0202

        n_test = round(min(x_train.shape[0] * 0.10, 5000))
        n_train = x_train.shape[0] - n_test
        x_test, y_test = x_train[n_train:], y_train[n_train:]
        x_train, y_train = x_train[:n_train], y_train[:n_train]
    elif dataset == "spect_10" or dataset == "spect_100":
        if dataset == "spect_10":
            subsize = 10000
        elif dataset == "spect_100":
            subsize = 100000

        df = pd.read_csv("dbs/spectroscopic.csv", ' ')
        ndf = np.random.permutation(df)
        y_train = np.array(ndf)[:subsize,-1:]
        x_train = np.array(ndf)[:subsize,:-1]

        n_test = round(min(x_train.shape[0] * 0.10, 5000))
        n_train = x_train.shape[0] - n_test
        x_test, y_test = x_train[n_train:], y_train[n_train:]
        x_train, y_train = x_train[:n_train], y_train[:n_train]

    if model == "rf":
        mobj = SKLFlexCodeRandomForest(n_estimators=20)
    elif model == "knn":
        mobj = SKLFlexCodeKNN()
        cv = ShuffleSplit(n_splits=1, test_size=n_test, random_state=0)
        gs_params = dict(
            k = np.array([1, 5, 15, 25, 35, 45, 55, 65, 75]),
        )
        mobj = GridSearchCV(mobj, gs_params, cv=cv, n_jobs=1)
    elif model == "xgb":
        mobj = SKLFlexCodeXGBoost()
        cv = ShuffleSplit(n_splits=1, test_size=n_test, random_state=0)
        gs_params = dict(max_depth = np.array([6, 12]))
        mobj = GridSearchCV(mobj, gs_params, cv=cv, n_jobs=1)
    elif model == "ann":
        ncomponents = 1000
        mobj = NNCDE(
            ncomponents=ncomponents,
            verbose=2,
            beta_loss_penal_exp=0.0,
            beta_loss_penal_base=0.0,
            nn_weight_decay=0.0,
            es=True,
            hidden_size=2000,
            num_layers=10,
            #gpu=False,
        )
        mobj = Pipeline([
            ('stand', StandardScaler()),
            ('nnf_obj', mobj)]
        )

    h = hashlib.new('ripemd160')
    h.update(pickle.dumps(x_train))
    h.update(pickle.dumps(y_train))
    filename = ("models_cache/mobj_" + model + "_" +
                h.hexdigest() + ".pkl")
    if not os.path.isfile(filename):
        print("Started working on file", filename)
        start_time = time.time()
        mobj.fit(x_train, y_train)
        elapsed_time = time.time() - start_time
        joblib.dump([mobj, elapsed_time], filename)
        print("Saved file", filename)
    else:
        mobj, elapsed_time = joblib.load(filename)
        print("Loaded file", filename)

    print("Dataset:", dataset)
    print("Model:", model)
    #print("Score (utility) on test:", mobj.score(x_test, y_test))
    print("Elapsed time:", elapsed_time)

    return mobj, elapsed_time

datasets = ["diamonds", "pnad", "spect_10", "spect_100"]
models = ["rf", "knn", "xgb", "ann"]

for dataset in datasets:
    for model in models:
        call_test(dataset, model)

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

from nn_flexcode import NNFlexCode, set_cache_dir
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import hashlib
import pickle
from sklearn.externals import joblib
import os

set_cache_dir("nn_flexcode_fs_cache", bytes_limit=30*2**30)

#Comment for non-deterministic results
np.random.seed(10)

n_train = 900
n_test = 800
x_dim = 45
ncomponents = 500

beta = stats.norm.rvs(size=x_dim, scale=0.2)
sigma = 0.8
beta0 = -.3
func = lambda x: (np.dot(beta, x))

x_train = stats.norm.rvs(scale=1, size=n_train*x_dim).reshape((n_train, x_dim))
y_train = np.apply_along_axis(func, 1, x_train)[:, None]
y_train += stats.norm.rvs(loc=beta0, scale=sigma, size=n_train)[:, None]

x_test = stats.norm.rvs(scale=1, size=n_test*x_dim).reshape((n_test, x_dim))
y_test = np.apply_along_axis(func, 1, x_test)[:, None]
y_test += stats.norm.rvs(loc=beta0, scale=sigma, size=n_test)[:, None]

y_train = np.array(y_train, dtype='f4')
y_train = torch.from_numpy(y_train)
y_train = F.sigmoid(y_train).numpy()

y_test = np.array(y_test, dtype='f4')
y_test = torch.from_numpy(y_test)
y_test = F.sigmoid(y_test).numpy()

def true_pdf_calc(x_pred, y_pred):
    logit_y_pred = - np.log(1/y_pred - 1)
    mu = np.dot(x_pred, beta) + beta0
    density = stats.norm.pdf(logit_y_pred, mu, sigma)
    density /= np.abs(y_pred - y_pred**2)
    return density

nnf_obj = NNFlexCode(
ncomponents=ncomponents,
verbose=2,
beta_loss_penal_exp=0.4,
beta_loss_penal_base=0.3,
nn_weights_loss_penal=0.1,
)

nnf_obj.fit(x_train, y_train)
#nnf_obj.move_to_cpu()
print("Score (utility) on train:", nnf_obj.score(x_train, y_train))
print("Score (utility) on test:", nnf_obj.score(x_test, y_test))

est_pdf = nnf_obj.predict([x_test, y_test])
true_pdf = true_pdf_calc(x_test, y_test)
sq_errors = (est_pdf - true_pdf)**2
print("Squared density errors for test:\n", sq_errors)
print("\nAverage squared density errors for test:\n", sq_errors.mean())

#gs_params = dict(
#ncomponents = np.arange(500, 10, -10),
#beta_loss_penal_exp = np.arange(0, 2, .1),
#beta_loss_penal_base = np.arange(0, 2, .1),
#nn_weights_loss_penal = np.arange(0, 2, .1),
#)

gs_params = dict(
ncomponents = np.arange(500, 10, -10),
beta_loss_penal_exp = stats.truncnorm(a=-1.1, b=np.inf, scale=0.5,
                                      loc=1.1),
beta_loss_penal_base = stats.truncnorm(a=0, b=np.inf, scale=0.5),
nn_weights_loss_penal = stats.truncnorm(a=0, b=np.inf, scale=2.0),
)

h = hashlib.new('ripemd160')
h.update(pickle.dumps(x_train))
h.update(pickle.dumps(y_train))
i = 0
gs_clf_list = []
for i in range(10):
    filename = ("nn_flexcode_fs_cache/model_" + h.hexdigest() + "_"
                + str(i) + ".pkl")
    if not os.path.isfile(filename):
        print("Started working on file", filename)
        gs_clf = RandomizedSearchCV(nnf_obj, gs_params, n_iter=10)
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


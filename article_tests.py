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

#Comment for non-deterministic results
np.random.seed(10)

n_train = 100_000
n_test = 800
x_dim = 50
ncomponents = 400

beta_mu = stats.norm.rvs(size=x_dim, scale=0.01)
beta_sigma = stats.norm.rvs(size=x_dim, scale=0.01)
sigma = 0.8
beta0_mu = -.3
beta0_sigma = .1

def func_mu(x):
    x_transf = x.copy()
    for i in range(0, x_dim-5, 5):
        x_transf[i] = np.abs(x[i])**1.3
        x_transf[i+1] = np.cos(x[i+1])
        x_transf[i+2] = x[i]*x[i+2]
        x_transf[i+4] = np.sqrt(np.abs(x[i+4]))
        x_transf[i+5] = x[i+5] * np.sin(x[i])
    return np.dot(beta_mu, x)

def func_sigma(x):
    return np.abs(np.dot(beta_sigma, x))

def true_pdf_calc(x_pred, y_pred):
    logit_y_pred = - np.log(1/y_pred - 1)
    mu = np.apply_along_axis(func_mu, 1, x_pred) + beta0_mu
    sigma = np.apply_along_axis(func_sigma, 1, x_pred) + beta0_sigma
    density = stats.skewnorm.pdf(logit_y_pred, loc=mu, scale=sigma, a=4)
    density /= np.abs(y_pred - y_pred**2)
    return density

def generate_data(n_gen):
    x_gen = stats.skewnorm.rvs(scale=4, size=n_gen*x_dim, a=2)
    x_gen = x_gen.reshape((n_gen, x_dim))

    mu_gen = np.apply_along_axis(func_mu, 1, x_gen)
    sigma_gen = np.apply_along_axis(func_sigma, 1, x_gen)

    y_gen = stats.skewnorm.rvs(loc=beta0_mu, scale=1,
                               size=n_gen, a=4)
    y_gen = mu_gen + y_gen * (sigma_gen + beta0_sigma)
    y_gen = y_gen * sigma_gen + mu_gen

    y_gen = np.array(y_gen[:, None], dtype='f4')
    y_gen = torch.from_numpy(y_gen)
    y_gen = F.sigmoid(y_gen).numpy()

    return x_gen, y_gen

x_train, y_train = generate_data(n_train)
x_test, y_test = generate_data(n_test)

nnf_obj = NNFlexCode(
ncomponents=ncomponents,
verbose=2,
#beta_loss_penal_exp=0.4,
#beta_loss_penal_base=0.3,
#nn_weights_loss_penal=0.1,
nhlayers=10,
es=True,
)

"""
nnf_obj.fit(x_train, y_train)
print("Score (utility) on train:", nnf_obj.score(x_train, y_train))
print("Score (utility) on test:", nnf_obj.score(x_test, y_test))

est_pdf = nnf_obj.predict([x_test, y_test])
true_pdf = true_pdf_calc(x_test, y_test)
sq_errors = (est_pdf - true_pdf)**2
print("Squared density errors for test:\n", sq_errors)
print("\nAverage squared density errors for test:\n", sq_errors.mean())
"""

cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

gs_params_list = [
  dict(
  ncomponents = np.arange(500, 30, -10),
  nhlayers = np.arange(1, 20, 3),
  ),
  dict(
  nn_weights_loss_penal = np.concatenate([np.logspace(-10, 0, 30),
                                          np.logspace(0, 1, 30)]),
  nhlayers = np.arange(1, 20, 3),
  ),
  dict(
  beta_loss_penal_exp = np.concatenate([np.logspace(-10, 0, 7),
                                          np.logspace(0, 1, 7)]),
  beta_loss_penal_base = np.concatenate([np.logspace(-10, 0, 7),
                                          np.logspace(0, 1, 7)]),
  nhlayers = np.arange(1, 20, 3),
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

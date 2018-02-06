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
from sklearn.model_selection import GridSearchCV

set_cache_dir("nn_flexcode_fs_cache")

#Comment for non-deterministic results
np.random.seed(10)

n_train = 100_000
n_test = 800
x_dim = 45
ncomponents = 100

beta = stats.norm.rvs(size=x_dim, scale=0.2)
func = lambda x: (np.dot(beta, x))

x_train = stats.norm.rvs(scale=1, size=n_train*x_dim).reshape((n_train, x_dim))
y_train = np.apply_along_axis(func, 1, x_train)[:, None]
y_train += stats.norm.rvs(loc=-.3, scale=.8, size=n_train)[:, None]

x_test = stats.norm.rvs(scale=1, size=n_test*x_dim).reshape((n_test, x_dim))
y_test = np.apply_along_axis(func, 1, x_test)[:, None]
y_test += stats.norm.rvs(loc=-.3, scale=.8, size=n_test)[:, None]

y_train = np.array(y_train, dtype='f4')
y_train = torch.from_numpy(y_train)
y_train = F.sigmoid(y_train).numpy()

y_test = np.array(y_test, dtype='f4')
y_test = torch.from_numpy(y_test)
y_test = F.sigmoid(y_test).numpy()

nnf_obj = NNFlexCode(ncomponents=ncomponents)
nnf_obj.fit(x_train, y_train)
nnf_obj.move_to_cpu()
print("Score (utility) on train:", nnf_obj.score(x_train, y_train))
print("Score (utility) on test:", nnf_obj.score(x_test, y_test))


gs_params = {'batch_initial': [200, 300]}
gs_clf = GridSearchCV(NNFlexCode(ncomponents=ncomponents), gs_params)

gs_clf.fit(x_train, y_train)
gs_clf.predict(x_test)
gs_clf.score(x_test, y_test)


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

from sklearn.model_selection import ShuffleSplit

import flexcode
from flexcode.regression_models import NN

from generate_data import generate_data, true_pdf_calc

n_train = 100_000
n_test = 800
x_train, y_train = generate_data(n_train)
x_test, y_test = generate_data(n_test)

splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
index_train, index_val = next(iter(splitter.split(x_train, y_train)))

x_val, y_val = x_train[index_val], y_train[index_val]
x_train, y_train = x_train[index_train], y_train[index_train]

# Parameterize model
model = flexcode.FlexCodeModel(NN, max_basis=30, basis_system="Fourier",
                               regression_params={"k":20}, z_min=0,
                               z_max=1)

# Fit and tune model
model.fit(x_train, y_train)
model.tune(x_val, y_val)

print("Score (utility) on test:",
      - model.estimate_error(x_test, y_test))

#Check using true density information
est_pdf, y_grid = model.predict(x_test, n_grid=1000)
est_pdf = est_pdf[:, 1:-1]
true_pdf = true_pdf_calc(x_test, y_grid[1:-1][:,None]).T
sq_errors = (est_pdf - true_pdf)**2
#print("Squared density errors for test:\n", sq_errors)
print("\nAverage squared density errors for test:\n", sq_errors.mean())

import matplotlib.pyplot as plt
plt.plot(true_pdf[1])
plt.plot(est_pdf[1])
plt.show()

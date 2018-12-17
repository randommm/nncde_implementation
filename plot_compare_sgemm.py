#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 2 or 3 the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from prepare_sgemm import get_sgemm_db
import hashlib
import pickle

np.random.seed(10)
df = get_sgemm_db()
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

preds = dict()
dataset = "sgemm"
models = ["ann_100", "rf"]
preprocess = "standard"

for model in models:
    h = hashlib.new('ripemd160')
    h.update(pickle.dumps(x_train))
    h.update(pickle.dumps(y_train))
    filename = "models_cache/"
    filename += "model_" + model
    filename += "_preprocess_" + preprocess
    filename += "_data_" + h.hexdigest()
    filename += ".pkl"
    mobj, elapsed_time = joblib.load(filename)
    print("Loaded file", filename)

    pred = mobj.predict(x_test[[0],])
    print("Dataset:", dataset)
    print("Model:", model)
    print("Preprocess:", preprocess)
    print("Elapsed time:", elapsed_time)
    preds[model] = pred

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, 1, len(preds["ann_100"])), preds["ann_100"], color="blue", label="ANN Fourier 100")
#ax.plot(np.linspace(0, 1, len(preds["ann_1000"])), preds["ann_1000"], color="red", label="ANN Fourier 1000", linestyle=".")
ax.plot(np.linspace(0, 1, len(preds["rf"])), preds["rf"], color="green", label="FC RF", linestyle="--")
ax.set_ylim(0, 100)
ax.set_yscale('symlog')

ax.set_xlabel("$y$")
ax.set_ylabel("$f(y)$")

legend = ax.legend()

with PdfPages("plots/compare_sgemm.pdf") as ps:
    ps.savefig(fig, bbox_inches='tight')

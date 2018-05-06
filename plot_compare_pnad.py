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

z_predicted_nn = joblib.load("z_pred_nn_pnad.pkl")
z_predicted_rf = joblib.load("z_pred_rf_pnad.pkl")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, 1, len(z_predicted_nn)), z_predicted_nn, color="blue", label="ANN Fourier")
ax.plot(np.linspace(0, 1, len(z_predicted_rf)), z_predicted_rf, color="green", label="FC RF", linestyle="--")

ax.set_xlabel("$y$")
ax.set_ylabel("$f(y)$")

legend = ax.legend()

with PdfPages("plots/compare_pnad.pdf") as ps:
    ps.savefig(fig, bbox_inches='tight')

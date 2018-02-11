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
from torch.autograd import Variable
import numpy as np
import collections
import pickle
from sklearn.externals import joblib

class _FourierSeries:
    def __init__(self):
        self._fourier_component = self._base_fourier_component
        self._fourier_component_nocache = self._base_fourier_component

    def __call__(self, x, ncomponents, nocache=False):
        """Calculate Fourier Series Expansion.

        Parameters
        ----------
        x : 1D numpy.array, list or tuple of numbers to calculate
            fourier series expansion
        ncomponents : int
            number of components of the series

        Returns
        ----------
        2D numpy.array where each line is the Fourier series expansion
        of each component of x.
        """
        x = np.array(x, ndmin=1, dtype=np.float32).ravel()
        results = np.array(np.empty((x.size, ncomponents)),
                           dtype=np.float32, order='F')

        if not nocache:
            _fourier_component = self._fourier_component
        else:
            _fourier_component = self._fourier_component_nocache
        for j in range(ncomponents):
            results[:, j] = _fourier_component(x, j)

        results = np.array(results, dtype=np.float32, order='C')

        return(results)

    def _base_fourier_component(self, x, j):
        if j % 2 == 0:
            k = j + 2
        else:
            k = j + 1
        results_row = np.sqrt(2) * np.sin(k * np.pi * x)
        return results_row

fourierseries = _FourierSeries()

class _CacheData():
    def __init__(self):
        self.cache_dir = None
        self.cache = None

cache_data = _CacheData()

def set_cache_dir(cachedir, bytes_limit=10*2**30):
    cache = joblib.Memory(cachedir=cachedir, bytes_limit=bytes_limit,
                          verbose=0).cache
    cache_data.cache = cache
    cache_data.cachedir = cachedir
    fsc_cached = cache(fourierseries._base_fourier_component)
    fourierseries._fourier_component = fsc_cached

def _np_to_var(arr, volatile=False):
    arr = np.array(arr, dtype='f4')
    arr = Variable(torch.from_numpy(arr), volatile=volatile)
    return arr

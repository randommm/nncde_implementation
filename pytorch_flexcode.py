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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy as sp
import scipy.stats as stats
import statsmodels.formula.api as sm
import time

def fourierseries(x, ncomponents):
    """Calculate Fourier Series Expansion.

    Parameters
    ----------
    x : 1D numpy.array, list or tuple of numbers to calculate
        fourier series expansion
    ncomponents : int
        number of components of the series

    Returns
    ----------
    2D numpy.array where each line is the Fourier series expansion of
    each component of x.
    """
    from numpy import sqrt, sin, cos, pi
    x = np.array(x, ndmin=1, dtype=np.float32)
    results = np.array(np.empty((x.size, ncomponents)),
                       dtype=np.float32)

    for i in range(x.size):
        for j in range(ncomponents):
            if j%2 == 0:
                results[i, j] = sqrt(2) * sin((j+2) * pi * x[i])
            else:
                results[i, j] = sqrt(2) * cos((j+1) * pi * x[i])

    return(results)

#Comment for non-deterministic results
np.random.seed(10)

n_train = 100_000
n_test = 800
x_dim = 45
ncomponents = 500

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

def np_to_var(arr):
    arr = np.array(arr, dtype='f4')
    arr = Variable(torch.from_numpy(arr))
    return arr

inputv = np_to_var(x_train)
target = np_to_var(fourierseries(y_train, ncomponents))

z_grid = np.linspace(0, 1, 1000, dtype=np.float32)
phi_grid = np.array(fourierseries(z_grid, ncomponents).T)
phi_grid = np_to_var(phi_grid).cuda()

class Net(nn.Module):
    def __init__(self, x_dim, ncomponents):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_dim, ncomponents * 2)
        self.fc2 = nn.Linear(ncomponents * 2, ncomponents)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(x_dim, ncomponents).cuda()

initial_params = list(net.parameters())
criterion = nn.MSELoss()

nepoch = 30

start_time = time.process_time()
for epoch in range(nepoch):
    batch_size = min(10_000, 100 + 300 * int(epoch**1.3))
    optimizer = optim.SGD(net.parameters(), lr=.01/(1+epoch))
    permutation = torch.randperm(target.shape[0])

    inputv_perm = inputv.data[permutation]
    target_perm = target.data[permutation]
    inputv_perm = Variable(inputv_perm.pin_memory())
    target_perm = Variable(target_perm.pin_memory())

    for i in range(0, target.shape[0] + batch_size, batch_size):
        if i < target.shape[0]:
            inputv_next = inputv_perm[i:i+batch_size].cuda(async=True)
            target_next = target_perm[i:i+batch_size].cuda(async=True)

        if i != 0:
            optimizer.zero_grad()
            output = net(inputv_this)
            #loss = criterion(output, target_this)
            loss = -2 * (output * target_this).sum(1).mean()
            loss += (Variable.mm(output, phi_grid)**2).mean()
            loss.backward()
            optimizer.step()
            #print("Epoch", epoch, "and batch", i-1, "done", flush=True)

        inputv_this = inputv_next
        target_this = target_next
    print("Epoch", epoch, "done", flush=True)

elapsed_time = time.process_time() - start_time
print("Elapsed time:", elapsed_time, flush=True)

params = list(net.parameters())

net.cpu()
phi_grid = phi_grid.cpu()
output_train = net(inputv)
#loss_train = criterion(output_train, target)
loss = -2 * (output_train * target).sum(1).mean()
loss += (Variable.mm(output_train, phi_grid)**2).mean()
print("loss_train:", loss.data.numpy()[0])

output_test = net(np_to_var(x_test))
target_test = np_to_var(fourierseries(y_test, ncomponents))
#loss_test = criterion(output_test, target_test)
loss_test = -2 * (output_test * target_test).sum(1).mean()
loss_test += (Variable.mm(output_test, phi_grid)**2).mean()

print("loss_test:", loss_test.data.numpy()[0])

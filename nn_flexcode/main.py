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
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------
from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
from sklearn.base import BaseEstimator

from .utils import fourierseries, _np_to_var, cache_data

class _BaseNNFlexCode(BaseEstimator):
    def __init__(self,
                 ncomponents=100,
                 beta_loss_penal_exp=0,
                 beta_loss_penal_base=0,
                 nn_weights_loss_penal=0,

                 nepoch=500,

                 batch_initial=100,
                 batch_step_multiplier=5.0,
                 batch_step_epoch_expon=20.0,
                 batch_max_size=10000,

                 lr_initial=1,
                 lr_step_multiplier=0.0001,
                 lr_step_epoch_expon=3.0,
                 lr_min_size=0,

                 loss_of_train_using_regression=True,
                 grid_size=2000,
                 gpu=True,
                 verbose=1,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

        self.gpu = self.gpu and torch.cuda.is_available()

    def fit(self, x_train, y_train):
        self.ncomponents = int(self.ncomponents)
        self.x_dim = x_train.shape[1]
        self._construct_neural_net()
        self.epoch_count = 0

        self.z_grid = np.linspace(0, 1, self.grid_size,
                                  dtype=np.float32)
        self.phi_grid = np.array(fourierseries(self.z_grid,
            self.ncomponents).T)
        self.phi_grid = _np_to_var(self.phi_grid)

        if self.gpu:
            self.move_to_gpu()

        n_attempts = 0
        while True:
            try:
                return self.continue_fit(x_train, y_train, self.nepoch)
            except RuntimeError as err:
                print("Warning: RuntimeError with message:", err)
                if self.epoch_count >= 1 or n_attempts >= 30:
                    raise err
                elif n_attempts >= 2:
                    self.lr_initial *= .5
                    print("Cutting lr_initial by half and retrying")
                else:
                    print("Retrying")
                self._construct_neural_net()
                if self.gpu:
                    self.move_to_gpu()
                n_attempts += 1

    def move_to_gpu(self):
        self.neural_net.cuda()
        self.phi_grid = self.phi_grid.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.neural_net.cpu()
        self.phi_grid = self.phi_grid.cpu()
        self.gpu = False

        return self

    def continue_fit(self, x_train, y_train, nepoch):
        criterion = nn.MSELoss()

        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)

        assert(self.lr_initial > 0)
        assert(self.lr_step_multiplier > 0)
        assert(self.lr_step_epoch_expon > 0)
        assert(self.lr_min_size >= 0)
        assert(self.beta_loss_penal_exp >= 0)
        assert(self.beta_loss_penal_base >= 0)

        inputv = _np_to_var(x_train)
        target = _np_to_var(fourierseries(y_train, self.ncomponents))

        batch_max_size = min(self.batch_max_size, x_train.shape[0])

        start_time = time.process_time()

        loss_vals = []
        batch_sizes = []
        for _ in range(nepoch):
            batch_size = int(min(batch_max_size,
                self.batch_initial +
                self.batch_step_multiplier *
                self.epoch_count**self.batch_step_epoch_expon))
            lr_val = max(self.lr_min_size,
                self.lr_initial /
                (1 + self.lr_step_multiplier
                 * self.epoch_count**self.lr_step_epoch_expon))
            optimizer = optim.SGD(self.neural_net.parameters(),
                                  lr=lr_val)
            permutation = torch.randperm(target.shape[0])

            inputv_perm = inputv.data[permutation]
            target_perm = target.data[permutation]
            if self.gpu:
                inputv_perm = inputv_perm.pin_memory()
                target_perm = target_perm.pin_memory()
            inputv_perm = Variable(inputv_perm)
            target_perm = Variable(target_perm)

            for i in range(0, target.shape[0] + batch_size, batch_size):
                if i < target.shape[0]:
                    inputv_next = inputv_perm[i:i+batch_size]
                    target_next = target_perm[i:i+batch_size]

                    if self.gpu:
                        inputv_next = inputv_next.cuda(async=True)
                        target_next = target_next.cuda(async=True)

                if i != 0:
                    optimizer.zero_grad()
                    output = self.neural_net(inputv_this)

                    # Main loss
                    if self.loss_of_train_using_regression:
                        loss = criterion(output, target_this)
                    else:
                        loss1 = -2 * (output * target_this).sum(1)
                        loss1 = loss1.mean()
                        loss2 = (Variable.mm(output, self.phi_grid)
                                 .sum(1)**2)
                        loss2 = loss2.mean()
                        loss = loss1 + loss2

                    # Penalize on betas
                    if self.beta_loss_penal_base != 0:
                        penal = output ** 2
                        if self.beta_loss_penal_exp != 0:
                            aranged = Variable(
                                loss.data.new(
                                    range(1, self.ncomponents + 1))
                                ** self.beta_loss_penal_exp
                                )
                            penal = penal * aranged
                        penal = penal.sum(1).mean()
                        penal = penal * self.beta_loss_penal_base
                        loss += penal

                    # Penalize on nn weights
                    if self.nn_weights_loss_penal != 0:
                        penal = self.neural_net.parameters()
                        penal = map(lambda x: (x**2).sum(), penal)
                        penal = Variable.cat(tuple(penal)).sum()
                        loss += penal * self.nn_weights_loss_penal

                    # Correction for last batch as it might be smaller
                    this_batch_size = inputv_this.shape[0]
                    if (batch_size > inputv_this.shape[0]):
                        loss *= this_batch_size / batch_size

                    np_loss = loss.data.cpu().numpy()[0]
                    if np.isnan(np_loss):
                        raise RuntimeError("Loss is NaN")

                    if self.verbose >= 2:
                        loss_vals.append(np_loss)
                        batch_sizes.append(this_batch_size)

                    loss.backward()
                    optimizer.step()

                inputv_this = inputv_next
                target_this = target_next
            if self.verbose >= 2:
                avgloss = np.average(loss_vals, weights=batch_sizes)
                print("Epoch", self.epoch_count, "done, train loss",
                      avgloss, flush=True)
            self.epoch_count += 1

        elapsed_time = time.process_time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def score(self, x_test, y_test):
        inputv = _np_to_var(x_test, volatile=True)
        target = _np_to_var(fourierseries(y_test, self.ncomponents),
                            volatile=True)

        if self.gpu:
            inputv = Variable(inputv.data.pin_memory(), volatile=True)
            target = Variable(target.data.pin_memory(), volatile=True)

        batch_size = min(self.batch_max_size, x_test.shape[0])

        loss_vals = []
        batch_sizes = []
        for i in range(0, target.shape[0] + batch_size, batch_size):
            if i < target.shape[0]:
                inputv_next = inputv[i:i+batch_size]
                target_next = target[i:i+batch_size]

                if self.gpu:
                    inputv_next = inputv_next.cuda(async=True)
                    target_next = target_next.cuda(async=True)

            if i != 0:
                output = self.neural_net(inputv_this)
                loss1 = -2 * (output * target_this).sum(1)
                loss1 = loss1.mean()
                loss2 = (Variable.mm(output, self.phi_grid).sum(1)**2)
                loss2 = loss2.mean()
                loss = loss1 + loss2
                loss_vals.append(loss.data.cpu().numpy()[0])
                batch_sizes.append(inputv_this.shape[0])

            inputv_this = inputv_next
            target_this = target_next

        return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict(self, x_pred, y_pred):
        inputv = _np_to_var(x_pred, volatile=True)
        target = _np_to_var(fourierseries(y_pred, self.ncomponents),
                            volatile=True)
        if self.gpu:
            inputv = inputv.cuda()
            target = target.cuda()
        x_output_pred = self.neural_net(inputv)
        output_pred = self.neural_net(inputv) * target
        output_pred = output_pred.sum(1) + 1
        return output_pred.data.cpu().numpy()

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, x_dim, ncomponents):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(x_dim, ncomponents)
                self.fc2 = nn.Linear(ncomponents, ncomponents)
                self.fc3 = nn.Linear(ncomponents, ncomponents)
                self.fc4 = nn.Linear(ncomponents, ncomponents)
                self.fc5 = nn.Linear(ncomponents, ncomponents)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x
        self.neural_net = NeuralNet(self.x_dim, self.ncomponents)

    def __getstate__(self):
        d = self.__dict__.copy()
        if "neural_net" in d.keys():
            d["neural_net_params"] = self.neural_net.state_dict()
            del(d["neural_net"])

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del(self.neural_net_params)
            if self.gpu:
                self.move_to_gpu()

class NNFlexCode(_BaseNNFlexCode):
    def fit(self, x_train, y_train):
        cache = cache_data.cache
        if cache is None:
            return super.fit(x_train, y_train)

        fit_cached = cache(self.fit_cacheable, ignore=["self"])
        new_self = fit_cached(x_train, y_train, self.get_params())
        self.__dict__ = new_self.__dict__
        return self

    def fit_cacheable(self, x_train, y_train, params):
        return super().fit(x_train, y_train)

    def score(self, x_test, y_test):
        cache = cache_data.cache
        if cache is None:
            return super.score(x_train, y_train)

        score_cached = cache(self.score_cacheable, ignore=["self"])
        return score_cached(x_test, y_test, self.get_params())

    def score_cacheable(self, x_test, y_test, params):
        return super().score(x_test, y_test)

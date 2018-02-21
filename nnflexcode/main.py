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
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit

from .utils import fourierseries, _np_to_var, cache_data

class NNFlexCode(BaseEstimator):
    """
    Estimate univariate density using Bayesian Fourier Series.
    This method only works with data the lives in
    [0, 1], however, the class implements methods to automatically
    transform user inputted data to [0, 1]. See parameter `transform`
    below.

    Parameters
    ----------
    ncomponents : integer
        Maximum number of components of the Fourier series
        expansion.

    beta_loss_penal_exp : integer
        Exponential term for penalizaing the size of beta's of the Fourier Series. This penalization occurs for training only (does not affect score method nor validation set if es=True).
    beta_loss_penal_base : integer
        Base term for penalizaing the size of beta's of the Fourier Series. This penalization occurs for training only (does not affect score method nor validation set if es=True).
    nn_weights_loss_penal : integer
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score method nor validation set if es=True).

    nhlayers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hls_multiplier : integer
        Multiplier for the size of the hidden layers of the neural network. If set to 1, then each of them will have ncomponents components. If set to 2, then 2 * ncomponents components, and so on.

    es : bool
        If true, then will split the training set into training and validation and calculate the validation internally on each epoch and check if the validation loss increases or not.
    es_validation_set : float
        Size of the validation set if es == True.
    es_give_up_after_nepochs : float
        Amount of epochs to try to decrease the validation loss before giving up and stoping training.
    es_splitter_random_state : float
        Random state to split the dataset into training and validation.

    nepoch : integer
        Number of epochs to run. Ignored if es == True.

    batch_initial : integer
        Initial batch size.
    batch_step_multiplier : float
        See batch_inital.
    batch_step_epoch_expon : float
        See batch_inital.
    batch_max_size : float
        See batch_inital.

    grid_size : integer
        Set grid size for calculating utility on score method.
    batch_test_size : integer
        Size of the batch for validation and score methods.
        Does not affect training efficiency, usefull when there's
        little GPU memory.
    gpu : bool
        If true, will use gpu for computation, if available.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """
    def __init__(self,
                 ncomponents=50,
                 beta_loss_penal_exp=0,
                 beta_loss_penal_base=0,
                 nn_weights_loss_penal=0,
                 nhlayers=1,
                 hls_multiplier=5,

                 es = True,
                 es_validation_set = 0.1,
                 es_give_up_after_nepochs = 20,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=100,
                 batch_step_multiplier=1.2,
                 batch_step_epoch_expon=1.1,
                 batch_max_size=400,

                 grid_size=10000,
                 batch_test_size=2000,
                 gpu=True,
                 verbose=1,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):
        self.gpu = self.gpu and torch.cuda.is_available()

        #if self.divide_batch_max_size_by_nlayers:
        #    self.batch_max_size_c = self.batch_max_size // (
        #                                 self.nhlayers + 1)
        #else:
        #    self.batch_max_size_c = self.batch_max_size

        self.ncomponents = int(self.ncomponents)
        max_penal = self.ncomponents ** self.beta_loss_penal_exp
        if max_penal > 1e4:
            new_val = np.log(1e4) / np.log(self.ncomponents)
            print("Warning: beta_loss_penal_exp is very large for "
                  "this amount of components (ncomponents).\n",
                  "Will automatically decrease it to", new_val,
                  "to avoid having the model blow up.")
            self.beta_loss_penal_exp = new_val

        self.x_dim = x_train.shape[1]
        self._construct_neural_net()
        self.epoch_count = 0

        if self.gpu:
            self.move_to_gpu()

        return self.improve_fit(x_train, y_train, self.nepoch)

    def move_to_gpu(self):
        self.neural_net.cuda()
        if hasattr(self, "phi_grid"):
            self.phi_grid = self.phi_grid.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.neural_net.cpu()
        if hasattr(self, "phi_grid"):
            self.phi_grid = self.phi_grid.cpu()
        self.gpu = False

        return self

    def improve_fit(self, x_train, y_train, nepoch):
        criterion = nn.MSELoss()

        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        assert(self.beta_loss_penal_exp >= 0)
        assert(self.beta_loss_penal_base >= 0)
        assert(self.nn_weights_loss_penal >= 0)

        assert(self.nhlayers >= 0)
        assert(self.hls_multiplier > 0)

        inputv_train = np.array(x_train, dtype='f4')
        target_train = np.array(fourierseries(y_train,
            self.ncomponents), dtype='f4')

        range_epoch = range(nepoch)
        if self.es:
            splitter = ShuffleSplit(n_splits=1,
                test_size=self.es_validation_set,
                random_state=self.es_splitter_random_state)
            index_train, index_val = next(iter(splitter.split(x_train,
                y_train)))

            inputv_val = inputv_train[index_val]
            target_val = target_train[index_val]
            inputv_val = np.ascontiguousarray(inputv_train)
            target_val = np.ascontiguousarray(target_train)

            inputv_train = inputv_train[index_train]
            target_train = target_train[index_train]
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count() # infty iterator


        batch_max_size = min(self.batch_max_size, x_train.shape[0])
        batch_test_size = min(self.batch_test_size, x_train.shape[0])

        start_time = time.process_time()

        optimizer = optim.Adadelta(self.neural_net.parameters())
        for _ in range_epoch:
            batch_size = int(min(batch_max_size,
                self.batch_initial +
                self.batch_step_multiplier *
                self.epoch_count ** self.batch_step_epoch_expon))

            permutation = np.random.permutation(target_train.shape[0])
            inputv_train = torch.from_numpy(inputv_train[permutation])
            target_train = torch.from_numpy(target_train[permutation])
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            self.neural_net.train()
            self._one_epoch("train", batch_size, inputv_train,
                target_train, optimizer, criterion, volatile=False)
            if self.es:
                self.neural_net.eval()
                avloss = self._one_epoch("test", batch_test_size,
                    inputv_val, target_val, optimizer, criterion,
                    volatile=True)
                if avloss <= self.best_loss_val:
                    self.best_loss_val = avloss
                    best_state_dict = self.neural_net.state_dict()
                    es_tries = 0
                    if self.verbose >= 2:
                        print("This is the lowest validation loss",
                              "so far.")
                else:
                    es_tries += 1
                if es_tries >= self.es_give_up_after_nepochs:
                    self.neural_net.load_state_dict(best_state_dict)
                    if self.verbose >= 1:
                        print("Validation loss did not improve after",
                              self.es_give_up_after_nepochs, "tries.")
                    break

            self.epoch_count += 1

        elapsed_time = time.process_time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def _one_epoch(self, ftype, batch_size, inputv, target, optimizer,
                   criterion, volatile):

        inputv = torch.from_numpy(inputv)
        target = torch.from_numpy(target)
        if self.gpu:
            inputv = inputv.pin_memory()
            target = target.pin_memory()
        inputv = Variable(inputv, volatile=volatile)
        target = Variable(target, volatile=volatile)

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
                batch_actual_size = inputv_this.shape[0]
                if batch_actual_size != batch_size and ftype == "train":
                    continue

                optimizer.zero_grad()
                output = self.neural_net(inputv_this)

                # Main loss
                loss = criterion(output, target_this)

                #loss1 = -2 * (output * target_this).sum(1)
                #loss1 = loss1.mean()
                #self._create_phi_grid()

                #loss2 = Variable.mm(output, self.phi_grid)**2
                #loss2 = loss2.mean()

                #loss = loss1 + loss2

                #alpha = min(0.1 * (self.epoch_count + 1) ** 2, 100)
                #loss = (output * target_this).sum(1) + 1
                #loss = F.softplus(loss, alpha)
                #loss = Variable.clamp(loss, 1e-30)
                #loss = - loss.log().mean()

                # Penalize on betas
                if self.beta_loss_penal_base != 0 and ftype == "train":
                    penal = output ** 2
                    if self.beta_loss_penal_exp != 0:
                        aranged = Variable(
                            loss.data.new(
                                range(1, self.ncomponents + 1))
                            ** self.beta_loss_penal_exp
                            )
                        penal = penal * aranged
                    penal = penal.mean()
                    penal = penal * self.beta_loss_penal_base
                    loss += penal

                # Penalize on nn weights
                if self.nn_weights_loss_penal != 0 and ftype == "train":
                    penal = self.neural_net.parameters()
                    penal = map(lambda x: (x**2).sum(), penal)
                    penal = Variable.cat(tuple(penal)).sum()
                    loss += penal * self.nn_weights_loss_penal

                # Correction for last batch as it might be smaller
                if batch_actual_size != batch_size:
                    loss *= batch_actual_size / batch_size

                np_loss = loss.data.cpu().numpy()[0]
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if ftype == "train":
                    loss.backward()
                    optimizer.step()

            inputv_this = inputv_next
            target_this = target_next

        avgloss = np.average(loss_vals, weights=batch_sizes)
        if self.verbose >= 2:
            print("Finished epoch", self.epoch_count,
                  "with batch size", batch_size,
                  "and", ftype,
                  ("pseudo-" if ftype == "train" else "") + "loss",
                  avgloss, flush=True)

        return avgloss

    def score(self, x_test, y_test):
        self.neural_net.eval()
        inputv = _np_to_var(x_test, volatile=True)
        target = _np_to_var(fourierseries(y_test, self.ncomponents),
                            volatile=True)

        if self.gpu:
            inputv = Variable(inputv.data.pin_memory(), volatile=True)
            target = Variable(target.data.pin_memory(), volatile=True)

        batch_size = min(self.batch_test_size, x_test.shape[0])

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

                #from scipy.integrate import quad
                #np_out = output.data.cpu().numpy()
                #def integrate_func(y):
                    #return (np_out_row *
                    #fourierseries(y, self.ncomponents, True)).sum()**2
                #loss2 = 0
                #for np_out_row in np_out:
                    #loss2 += quad(integrate_func, 0, 1, limit=1000)[0]
                #loss2 = loss2 / np_out.shape[0]
                #print(loss2)

                #for grid_size in range(1000, 50000, 1000):
                    #self.grid_size = grid_size
                    #print("for grid_size", self.grid_size)
                    #if not hasattr(self, "phi_grid"):
                        #self.y_grid = np.linspace(0, 1, self.grid_size,
                                             #dtype=np.float32)
                        #self.phi_grid = np.array(fourierseries(self.y_grid,
                                                 #self.ncomponents).T)
                        #self.phi_grid = _np_to_var(self.phi_grid)
                        #if self.gpu:
                            #self.phi_grid = self.phi_grid.cuda()

                    #loss2 = (Variable.mm(output, self.phi_grid).sum(1)**2)
                    #loss2 = loss2.mean()
                    #del(self.phi_grid)
                    #print(loss2)

                self._create_phi_grid()

                loss2 = Variable.mm(output, self.phi_grid)**2
                loss2 = loss2.mean()

                loss = loss1 + loss2
                loss_vals.append(loss.data.cpu().numpy()[0])
                batch_sizes.append(inputv_this.shape[0])

            inputv_this = inputv_next
            target_this = target_next

        return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict(self, x_pred, y_pred=None):
        self.neural_net.eval()
        inputv = _np_to_var(x_pred, volatile=True)
        if y_pred is None:
            self._create_phi_grid()
            target = self.phi_grid
        else:
            target = _np_to_var(fourierseries(y_pred, self.ncomponents),
                                volatile=True)
        if self.gpu:
            inputv = inputv.cuda()
            target = target.cuda()
        x_output_pred = self.neural_net(inputv)
        if y_pred is None:
            output_pred = Variable.mm(x_output_pred, target)
        else:
            output_pred = x_output_pred * target
            output_pred = output_pred.sum(1)
        output_pred += 1
        output_pred = F.relu(output_pred)
        return output_pred.data.cpu().numpy()

    def change_grid_size(self, new_grid_size):
        self.grid_size = new_grid_size
        if hasattr(self, "phi_grid"):
            del(self.phi_grid)
            del(self.y_grid)
            self._create_phi_grid()

    def _create_phi_grid(self):
        if not hasattr(self, "phi_grid"):
            self.y_grid = np.linspace(0, 1, self.grid_size,
                                      dtype=np.float32)
            self.phi_grid = np.array(fourierseries(self.y_grid,
                                     self.ncomponents).T)
            self.phi_grid = _np_to_var(self.phi_grid)
            if self.gpu:
                self.phi_grid = self.phi_grid.cuda()

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, x_dim, ncomponents, nhlayers,
                         hls_multiplier):
                super(NeuralNet, self).__init__()

                next_input_l_size = x_dim
                output_hl_size = int(ncomponents * hls_multiplier)
                self.m = nn.AlphaDropout(p=0.5)

                for i in range(nhlayers):
                    lname = "fc_" + str(i)
                    lnname = "fc_n_" + str(i)
                    self.__setattr__(lname,
                        nn.Linear(next_input_l_size, output_hl_size))
                    self.__setattr__(lnname,
                        nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(self.__getattr__(lname))

                self.fc_last = nn.Linear(next_input_l_size, ncomponents)
                self._initialize_layer(self.fc_last)
                self.exp_decay = nn.Parameter(torch.Tensor([0.0]))
                self.base_decay = nn.Parameter(torch.Tensor([-5.0]))

                self.nhlayers = nhlayers
                self.ncomponents = ncomponents
                self.np_sqrt2 = np.sqrt(2)

            def _decay_x(self, x):
                exp_decay = - Variable.exp(self.exp_decay)
                base_decay = Variable.exp(self.base_decay) + 1

                decay = Variable(
                    x.data.new(
                        range(1, self.ncomponents + 1))
                    ) * exp_decay
                decay = base_decay ** decay

                return x * decay

            def forward(self, x):
                for i in range(self.nhlayers):
                    fc = self.__getattr__("fc_" + str(i))
                    fcn = self.__getattr__("fc_n_" + str(i))
                    x = fcn(F.relu(fc(x)))
                    self.m(x)
                x = self.fc_last(x)
                x = F.sigmoid(x) * 2 * self.np_sqrt2 - self.np_sqrt2
                #x = self._decay_x(x)
                return x

            def _initialize_layer(self, layer):
                nn.init.constant(layer.bias, 0)
                gain=nn.init.calculate_gain('relu')
                nn.init.xavier_normal(layer.weight, gain=gain)

        self.neural_net = NeuralNet(self.x_dim, self.ncomponents,
                                    self.nhlayers, self.hls_multiplier)

    def __getstate__(self):
        d = self.__dict__.copy()
        if hasattr(self, "neural_net"):
            state_dict = self.neural_net.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].cpu()
            d["neural_net_params"] = state_dict
            del(d["neural_net"])

        #Delete phi_grid (will recreate on load)
        if hasattr(self, "phi_grid"):
            del(d["phi_grid"])
            d["y_grid"] = None

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del(self.neural_net_params)
            if self.gpu:
                self.move_to_gpu()

        #Recreate phi_grid
        if "y_grid" in d.keys():
            del(self.y_grid)
            self._create_phi_grid()

class NNFlexCodeCached(NNFlexCode):
    def fit(self, x_train, y_train):
        cache = cache_data.cache
        if cache is None:
            raise("Must set cache first!")
            # return super.fit(x_train, y_train)

        fit_cached = cache(self.fit_cacheable, ignore=["self"])
        new_self = fit_cached(x_train, y_train, self.get_params())
        self.__dict__ = new_self.__dict__
        return self

    def fit_cacheable(self, x_train, y_train, params):
        return super().fit(x_train, y_train)

    def score(self, x_test, y_test):
        cache = cache_data.cache
        if cache is None:
            raise("Must set cache first!")
            # return super.score(x_train, y_train)

        score_cached = cache(self.score_cacheable, ignore=["self"])
        return score_cached(x_test, y_test, self.get_params())

    def score_cacheable(self, x_test, y_test, params):
        return super().score(x_test, y_test)

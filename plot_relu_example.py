import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)
        self._initialize_layer(self.fc1)
        self._initialize_layer(self.fc2)
        self._initialize_layer(self.fc3)
        self._initialize_layer(self.fc4)
        self._initialize_layer(self.fc5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def _initialize_layer(self, layer):
        nn.init.normal(layer.bias, 0, 5)
        nn.init.uniform(layer.weight, -5, 5)



x_grid = np.linspace(-1, 1, 10000, dtype="f4")
x_grid_v = Variable(torch.from_numpy(x_grid[:,None]), volatile=True)
ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
for i in range(10):
    net = NeuralNet()
    ax.plot(x_grid, net(x_grid_v).data.numpy())

ax.set_xlabel("input")
ax.set_ylabel("output")

with PdfPages("plots/relu_example.pdf") as ps:
    ps.savefig(ax.get_figure(), bbox_inches='tight')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

cls = ["-", ":", "-.", "-", "--", "-."]
clw = [2.0, 2.0, 2.0, 2.0, 2.5, 1.0]

x_grid = np.linspace(-100, 100, 10000, dtype="f4")
x_grid_v = Variable(torch.from_numpy(x_grid[:,None]), volatile=True)

output_softplus = F.softplus(x_grid_v).data.numpy()
output_exp = Variable.exp(x_grid_v).data.numpy()

ax = plt.figure().add_subplot(111)
ax.plot(x_grid, output_softplus, label="softplus",
        linestyle=cls[0], lw=clw[0])
ax.plot(x_grid, output_exp, label="exponential",
        linestyle=cls[1], lw=clw[1])
ax.plot(x_grid, x_grid, label="identity",
        linestyle=cls[2], lw=clw[2])

ax.set_xlabel("input")
ax.set_ylabel("output")
ax.set_ylim([-2, 7])
ax.set_xlim([-2, 7])

legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)

with PdfPages("plots/softplus_example.pdf") as ps:
    ps.savefig(ax.get_figure(), bbox_inches='tight')

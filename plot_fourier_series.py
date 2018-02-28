from numpy import cos, pi, sqrt, linspace, ones
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

cls = ["-", ":", "-.", "-", "--", "-."]
clw = [1.0, 2.0, 2.0, 2.0, 2.5, 1.0]

ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
grid = linspace(0, 1, 10000)

func = lambda t: ones(t.shape)
for i in range(4):
    ax.plot(grid, func(grid), label="$\phi_{" + str(i*2) + "}$",
            linestyle=cls[i], lw=clw[i])
    i_next = i + 1
    func = lambda t: sqrt(2) * cos(i_next * 2 * pi * t)

ax.set_xlim([-0.03, 1.03])

legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

ax.set_xlabel("$y$")
ax.set_ylabel("$\phi_i(y)$")

with PdfPages("plots/fourier_components.pdf") as ps:
    ps.savefig(ax.get_figure(), bbox_inches='tight')

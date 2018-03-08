import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

G = nx.DiGraph()
node_list = []
node_list.append(['$X_1^{(1)}$', '$X_2^{(1)}$'])
node_list.append(['$X_1^{(2)}$', '$X_2^{(2)}$', '$X_3^{(2)}$',
                  '$X_4^{(2)}$'])
node_list.append(['$X_1^{(3)}$', '$X_2^{(3)}$'])
node_list.append(['$X_1^{(4)}$', '$X_2^{(4)}$', '$X_3^{(4)}$'])


betas = {}
edge_to_draw = []
for l in [0, 1, 2]:
    for i, nin in enumerate(node_list[l]):
        for o, nout in enumerate(node_list[l + 1]):
            cond1 = l == 0 and o == 1
            cond2 = l == 1 and i == 1
            cond3 = l == 1 and o == 1
            cond4 = l == 2 and i == 1
            if not (cond1 or cond2 or cond3 or cond4):
                edge_to_draw.append((nin, nout))
            G.add_edges_from([(nin, nout)])
            betas[(nin, nout)] = ('$\\beta_{'+str(i+1)+','+str(o+1)+
                                  '}^{('+str(l+1)+')}$')

pos = {}
horizontal = -1
for layer in node_list:
    vdenom = len(layer) + 1
    vnumer = len(layer)
    for node in layer:
        pos[node] = np.array([horizontal, 1/vdenom * vnumer])
        vnumer -= 1
    horizontal += 1

colors = {}
for v, layer in enumerate(node_list):
    for node in layer:
        colors[node] = v
colors = [colors[node] for node in G.nodes()]

ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
nodes = nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                       node_color = colors, node_size = 1200, alpha=0.3)
nx.draw_networkx_labels(G, pos)
edges = nx.draw_networkx_edges(G, pos, edge_to_draw, arrows=True,
    arrowstyle=ArrowStyle("Fancy", head_length=2.4,
                          head_width=1.0, tail_width=0.1))
#nx.draw_networkx_edge_labels(G, pos, edge_labels=betas)
plt.axis('off')

for edge in edges:
    edge.shrinkA=20
    edge.shrinkB=20

with PdfPages("plots/nodes_nn_dropout.pdf") as ps:
    ps.savefig(ax.get_figure(), bbox_inches='tight')

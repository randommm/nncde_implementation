import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

G = nx.DiGraph()
nodes_input = ['$X_1^{(2)}$', '$X_2^{(2)}$', '$X_3^{(2)}}$',
               '$X_4^{(2)}}$']
nodes_weights = ['$\\beta^{(2)}_{1,1}$', '$\\beta^{(2)}_{2,1}$',
               '$\\beta^{(2)}_{3,1}$', '$\\beta^{(2)}_{4,1}$']
nodes_mult = ['$\\times$', '$\\times $', '$ \\times$', '$ \\times $']

node_sum = '+'
node_intercept = '$\\beta^{(2)}_{0,1}$'
node_output = '$X_1^{(3)}$'
node_actf = '$\\mathbb{F}^{(2)}$'


pos = {}
pos[nodes_input[0]] = 4
pos[nodes_input[1]] = 2.4
pos[nodes_input[2]] = 1.6
pos[nodes_input[3]] = 0

pos[nodes_weights[0]] = (4*2.3 + 2.4) / 3.3
pos[nodes_weights[1]] = (4 + 2.4*2.3) / 3.3
pos[nodes_weights[2]] = (1.6*2.3) / 3.3
pos[nodes_weights[3]] = (1.6) / 3.3

pos[node_sum] = np.array([3, 2])
pos[node_intercept] = np.array([3.5, 4])
pos[node_actf] = np.array([4, 2])
pos[node_output] = np.array([5, 2])

for i in range(4):
    addt = .3 if i == 0 or i == 3 else 0
    pos[nodes_mult[i]] = np.array([2 + addt, pos[nodes_input[i]]])
    pos[nodes_input[i]] = np.array([0, pos[nodes_input[i]]])
    pos[nodes_weights[i]] = np.array([1, pos[nodes_weights[i]]])

    G.add_edges_from([(nodes_input[i], nodes_mult[i])])
    G.add_edges_from([(nodes_weights[i], nodes_mult[i])])
    G.add_edges_from([(nodes_mult[i], node_sum)])

G.add_edges_from([(node_intercept, node_sum)])
G.add_edges_from([(node_sum, node_actf)])
G.add_edges_from([(node_actf, node_output)])

colors = []
for node in G.nodes():
    if node in nodes_input:
        colors.append('r')
    elif node in nodes_weights or node == node_intercept:
        colors.append('g')
    elif node in nodes_mult or node == node_sum or node == node_actf:
        colors.append('y')
    elif node == node_output:
        colors.append('b')


ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                       node_size = 1200, alpha=0.6, node_color=colors)
nx.draw_networkx_labels(G, pos)
edges = nx.draw_networkx_edges(G, pos, arrows=True,
                       arrowstyle=ArrowStyle("Fancy", head_length=2.4,
                       head_width=1.0, tail_width=0.1))
#nx.draw_networkx_edge_labels(G, pos, edge_labels=betas)
for edge in edges:
    edge.shrinkA=20
    edge.shrinkB=20

plt.axis('off')
with PdfPages("plots/nodes_nn_detailed.pdf") as ps:
    ps.savefig(ax.get_figure(), bbox_inches='tight')

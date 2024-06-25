import networkx as nx
import numpy as np
from networkx import from_numpy_array, DiGraph

import notears.utils as notears_utils


def apply_threshold(W, w_threshold):
    W_t = np.copy(W)
    W_t[np.abs(W) < w_threshold] = 0
    return W_t


def compute_shd(W_true, W_est):
    acc = notears_utils.count_accuracy(W_true != 0, W_est != 0)
    return acc['shd'], acc


def find_minimal_dag_threshold(W):
    if notears_utils.is_dag(W):
        return 0, W
    possible_thresholds = sorted((abs(t) for t in W.flatten() if abs(t) > 0))
    for t_candidate in possible_thresholds:
        W[np.abs(W) < t_candidate] = 0
        if notears_utils.is_dag(W):
            return t_candidate, W
    assert False  # Should always find a dag


def find_optimal_threshold(W_true, W_est):
    possible_thresholds = sorted((abs(t) for t in W_est.flatten() if abs(t) > 0))
    best_t = max(possible_thresholds) if possible_thresholds else 0
    best_shd = W_true.shape[0]**2
    _, best_acc = compute_shd(W_true, W_est)
    best_W = W_est
    for t_candidate in possible_thresholds:
        W_est_t = apply_threshold(W_est, t_candidate)
        shd, acc = compute_shd(W_true, W_est_t)
        if shd < best_shd:
            best_t = t_candidate
            best_shd = shd
            best_acc = acc
            best_W = W_est_t
    return best_t, best_shd, best_W, best_acc


def least_square_cost(X, W):
    n, d = X.shape
    val = sum((X[i,j] - sum(X[i, k] * W[k, j] for k in range(d) if k != j))**2 for i in range(n) for j in range(d))
    return val


def plot(W, filename=None):
    import matplotlib.pyplot as plt
    # if abbrev:
    #     ls = dict((x,x[:3]) for x in self.nodes)
    # else:
    #     ls = None
    # try:
    #     edge_colors = [self._edge_colour[compelled] for (u,v,compelled) in self.edges.data('compelled')]
    # except KeyError:
    #     edge_colors = 'k'
    graph = from_numpy_array(W, create_using=DiGraph)
    fig, ax = plt.subplots()
    nx.draw_networkx(graph, ax=ax, pos=nx.drawing.nx_agraph.graphviz_layout(graph,prog='dot'),
                     node_color="white",arrowsize=15)
    if filename is not None:
        fig.savefig(filename,format='png', bbox_inches='tight')
    else:
        plt.show()

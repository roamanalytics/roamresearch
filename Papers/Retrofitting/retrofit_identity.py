import numpy as np

def retrofit_identity(X, edges, n_iter=100, alpha=None, beta=None, tol=1e-2,
                      verbose=False):
    """ Implements the baseline retrofitting method of Faruqui et al.

    Parameters
    ----------
    X : np.array (distributional embeddings)
    edges : dict containing edges. If multiple types of edges,
        this will be flattened.
    n_iter : int indicating the maximum number of iterations to run.
    alpha : func from `edges.keys()` to floats or None
    beta : func from `edges.keys()` to floats or None
    tol : float
        If the average distance change between two rounds is at or
        below this value, we stop. Default to 10^-2 as suggested
        in the paper.

    Returns
    -------
    Y : np.array, same dimensions and arrangement as `X`.
    """
    if isinstance(next(iter(edges.values())), dict):
        edges = flatten_edges(edges, len(X))

    if not alpha:
        alpha = lambda x: 1
    if not beta:
        beta = lambda x: 1 / len(edges[x])
    
    Y = X.copy()
    Y_prev = Y.copy()
    for iteration in range(1, n_iter+1):
        if verbose:
            print("Iteration {} of {}".format(iteration, n_iter), end='\r')
        for i, vec in enumerate(X):
            neighbors = edges[i]
            n_neighbors = len(neighbors)
            if n_neighbors:
                a = alpha(i)
                b = beta(i)
                retro = np.array([b * Y[j] for j in neighbors])
                retro = retro.sum(axis=0) + (a * X[i])
                norm = np.array([b for j in neighbors])
                norm = norm.sum(axis=0) + a
                Y[i] = retro / norm
        changes = np.abs(np.mean(np.linalg.norm(
            np.squeeze(Y_prev)[:1000] - np.squeeze(Y)[:1000], ord=2)))
        if changes <= tol:
            if verbose:
                print("Converged at iteration {}".format(iteration))
            return Y
        else:
            Y_prev = Y.copy()
    if verbose:
        print("Stopping at iteration {:d}; change was {:.4f}".format(iteration, changes))
    return Y


def flatten_edges(edges, n_nodes):
    """ Flattens a dict of dict of edges of different types.

    Parameters
    ----------
    edges   : dict that maps edge type to dict that maps index to neighbors.
    n_nodes : the number of nodes in the graph.

    Returns
    -------
    edges_naive : dict that maps index to all neighbors
    """
    edges_naive = {}
    for i in range(n_nodes):
        edges_naive[i] = []
        for rel_name in edges.keys():
            edges_r = edges[rel_name]
            try:
                my_edges = edges_r[i]
            except KeyError:
                continue
            edges_naive[i].extend(my_edges)
    return edges_naive
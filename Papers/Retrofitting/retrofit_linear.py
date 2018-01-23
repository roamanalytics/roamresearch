import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import normalize

def retrofit_linear(X, in_edges, out_edges, n_iter=10, alpha=None, beta=None,
                    tol=1e-2, lr=1.0, lr_decay=0.9, lam=1e-5, verbose=False,
                    A=None, orthogonal=True):
    """ Retrofit according to the linear mapping. See Eq (2) of the paper.

    Parameters
    ----------
    X          : np.array (distributional embeddings)
    in_edges   : dict that maps edge type to dict that maps entity index
                 to list of neighbors by an incoming edge
    out_edges  : dict that maps edge type to dict that maps entity index
                 to list of neighbors by an outgoing edge
    n_iter     : int indicating the maximum number of iterations
    alpha      : func from `edges[i].keys()` to floats or None
    beta       : func from `edges[i].keys()` to floats or None
    tol        : float
                 If the average distance change between two rounds is at or
                 below this value, we stop. Default to 10^-2 as suggested
                 in the paper.
    lr         : float learning rate
    lr_decay   : float learning rate decay
    lam        : float L2 regularization coefficient
    verbose    : bool indicating whether to print intermediate results
    A          : dict that maps edge type to np.array
                 If a linear mapping is known a priori, setting this value
                 will enforce it.
    orthogonal : bool indicating whether A should be constrained to be orthogonal.

    Returns
    -------
    Y : np.array, same dimensions and arrangement as `X`.
    A : dict that maps edge_type to an np.array, see eq(2)
    B : dict that maps edge_type to an np.array, see eq(2)
    """
    n_relation_types = len(in_edges)

    if not alpha:
        alpha = lambda i: 1
    if not beta:
        beta = lambda i,j,r: 1 / max(
            [np.sum([len(er[i]) for er in in_edges.values()]), 1]
            )*(int(j in in_edges[r][i]))
    if A is None:
        should_calc_A = True
    else:
        should_calc_A = False
    X = np.expand_dims(X, axis=2)
    Y = X.copy()
    Y_prev = Y.copy()
    n_nodes = len(X)
    # Intialize A_prev and B_prev.
    # Need to check for an example relationship to discover dimensionality.
    A_prev = {}
    B_prev = {}
    for rel in in_edges.keys():
        A_prev[rel] = np.zeros((1,1))
        B_prev[rel] = np.zeros((1,1))
        for i, neighbors in in_edges[rel].items():
            if len(neighbors) > 0:
                j = neighbors[0]
                d1 = Y[i].shape[0]
                d2 = Y[j].shape[0]
                A_prev[rel] = np.zeros((d1, d2))
                B_prev[rel] = np.zeros((d1, 1))
                break
            if i == len(neighbors) - 1:
                print("A[{}] has 0 edges.".format(r))

    # Sample negative edges.
    neg_edges = {r: {} for r in in_edges.keys()}
    neg_out_edges = {r: {i : [] for i in range(n_nodes)} for r in in_edges.keys()}
    for r, in_edges_r in in_edges.items():
        for i, neighbors in in_edges_r.items():
            # Not quite guaranteed to be non-neighbors, but is whp as n_nodes -> infty.
            neg_edges[r][i] = np.random.choice(n_nodes, size=len(neighbors))
            for j in neg_edges[r][i]:
                neg_out_edges[r][j].append(i)

    prev_loss = np.inf
    for iteration in range(1, n_iter+1):
        if verbose:
            print("Iteration {} of {}".format(iteration, n_iter), end='\r')
            print("Calculating B...", end='\r')

        B = calc_B(Y_prev, A_prev, beta, in_edges, neg_edges)
        B = {r: B_prev[r] + lr*(B[r]-B_prev[r]) for r in in_edges.keys()}
        if should_calc_A:
            if verbose:
                print("Calculating A...", end='\r')
            try:
                A = calc_A(Y, B, beta, in_edges, neg_edges, lam, orthogonal=True)
                A = {r: A_prev[r] + lr*(A[r]-A_prev[r]) for r in in_edges.keys()}
            except np.linalg.LinAlgError:
                print("SVD did not converge. Try a smaller lr.")
                return Y_prev, A_prev, B_prev
        if verbose:
            print("Calculating Y...", end='\r')
        Y = calc_Y(X, Y, A, B, in_edges, out_edges, neg_edges, neg_out_edges,
                   alpha, beta)
        Y = Y_prev + lr*(Y-Y_prev)

        if np.any(np.any(np.isnan(Y))):
            print("Y Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev), A_prev, B_prev
        if np.any([np.any(np.any(np.isnan(A[r]))) for r in in_edges.keys()]):
            print("A Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev), A_prev, B_prev
        if np.any([np.any(np.isnan(B[r])) for r in in_edges.keys()]):
            print("B Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev), A_prev, B_prev

        loss = calc_loss(X, Y, A, B, alpha, beta, lam, in_edges, neg_edges)
        if loss > prev_loss:
            print("Loss reached local minimum at iteration {}".format(iteration-1))
            return np.squeeze(Y_prev), A_prev, B_prev

        prev_loss = loss
        changes = np.mean(np.abs(np.linalg.norm(
            np.squeeze(Y_prev[:1000]) - np.squeeze(Y[:1000]), ord=2)))
        if verbose:
            print("Iteration {:d} of {:d}\tChanges: {:.3f}\tLoss: {:.3f}".format(iteration, n_iter, changes, loss))
        if changes <= tol:
            print("Converged at iteration {}".format(iteration))
            return np.squeeze(Y), A, B
        else:
            Y_prev = Y.copy()
            A_prev = A.copy()
            B_prev = B.copy()
            lr *= lr_decay

    print("Stopping at iteration {:d}; change was {:.3f}".format(iteration, changes))
    return np.squeeze(Y), A, B


def calc_Ar(Y, b_r, beta, edges, neg_edges, lam, orthogonal=True):
    """ Calculate a new A for a single edge type r according to Equation (5)

    Parameters
    ----------
    Y          : np.array (distributional embeddings)
    b_r        : np.array, bias term for this edge type
    beta       : func from 'edges.keys()' to float
    edges      : dict that maps entity index to list of neighbors
    neg_edges  : dict that maps entity index to list of non-neighbors
    lam        : float regularization parameter
    orthogonal : bool indicating whether Ar should be orthogonal.

    Returns
    -------
    A_r : np.array
    """

    # Get dimensionality
    for i, neighbors in edges.items():
        if len(neighbors) > 0:
            d1 = Y[i].shape[0]
            d2 = Y[neighbors[0]].shape[0]
            break

    term1 = np.zeros((d1, d2))
    term2 = lam*np.eye(d2)
    for i, neighbors in edges.items():
        for j in neighbors:
            term1 += beta(i,j)*(Y[i]-b_r).dot(Y[j].T)
            term2 += beta(i,j)*Y[j].dot(Y[j].T)
    for i, neighbors in neg_edges.items():
        for j in neighbors:
            try:
                term1 -= beta(i,j)*(Y[i]-b_r).dot(Y[j].T)
                term2 -= beta(i,j)*Y[j].dot(Y[j].T)
            except LinAlgError:
                # It is possible that a non-edge has incorrect dimensionality.
                continue
    A_r = term1.dot(np.linalg.inv(term2))
    if orthogonal:
        sq = np.asmatrix(sqrtm(A_r.T.dot(A_r)))
        A_r = np.real(A_r.dot(np.linalg.pinv(sq)))  # nearest orthogonal matrix
    return A_r


def calc_A(Y, B, beta, edges, neg_edges, lam=0., orthogonal=True):
    """ Calculate a new A value for each edge type.

    Parameters
    ----------
    Y          : np.array (distributional embeddings)
    b_r        : np.array, bias term for this edge type
    beta       : func from 'edges.keys()' to float
    edges      : dict that maps edge type to dict that maps entity index to list of neighbors
    neg_edges  : dict that maps edge type to dict that maps entity index to list of non-neighbors
    lam        : float regularization parameter
    orthogonal : bool indicating whether Ar should be orthogonal.

    Returns
    -------
    dict that maps edge type to A_r
    """
    return {r: calc_Ar(Y, B[r], lambda i,j: beta(i,j,r), edges[r],
                       neg_edges[r], lam, orthogonal)
            for r in edges.keys()}


def calc_Y(X, Y, A, b, in_edges, out_edges, neg_in_edges, neg_out_edges,
           alpha, beta):
    """ Calculates a new embedding based on Eq 6 of the paper.

    Parameters
    ----------
    X             : np.array, distributional embeddings
    Y             : np.array, current estimate of embeddings
    A             : dict that maps edge type to np array of linear mapping
    b             : dict that maps edge type to np array of bias vector
    in_edges      : dict that maps edge type to dict that maps entity index to list of neighbors
    out_edges     : dict that maps edge type to dict that maps entity index to list of neighbors
    neg_in_edges  : dict that maps edge type to dict that maps entity index to list of non-neighbors
    neg_out_edges : dict that maps edge type to dict that maps entity index to list of non-neighbors
    alpha         : func from 'edges.keys()' to float
    beta          : func from 'edges.keys()' to float

    Returns
    -------
    Y : np.array, new embeddings
    """
    for i, vec in enumerate(X):
        if i % 1000 == 0:
            print("i:{}".format(i), end='\r')
        numerator = alpha(i)*vec
        denominator = alpha(i)
        for r in in_edges.keys():
            for j in in_edges[r][i]:
                numerator += beta(i,j,r)*(A[r].dot(Y[j]) + b[r])
                denominator += beta(i,j,r)

            for j in out_edges[r][i]:
                numerator += beta(j,i,r)*(A[r].T.dot(Y[j] - b[r]))
                denominator += beta(j,i,r)

            for j in neg_in_edges[r][i]:
                numerator -= beta(i,j,r)*(A[r].dot(Y[j]) + b[r])
                denominator -= beta(i,j,r)

            for j in neg_out_edges[r][i]:
                numerator -= beta(j,i,r)*(A[r].T.dot(Y[j] - b[r]))
                denominator -= beta(j,i,r)
        Y[i] = numerator / denominator
    Y = np.squeeze(Y)
    Y = normalize(Y, norm='l2')
    Y = np.expand_dims(Y, axis=2)
    return Y


def calc_Br(Y, A_r, beta, in_edges_r, neg_edges_r):
    """ Calculates a new bias vector for a single edge type according to Eq 4.

    Parameters
    ----------
    Y           : np.array, entity embeddings
    A_r         : np.array, linear mapping for this edge type
    beta        : func from in_edges.keys() to float
    in_edges_r  : dict of incoming edges for this edge type
    neg_edges_r : dict of incoming non-edges for this edge type

    Returns
    -------
    np.array
    """
    num = 0.
    denom = 0.
    for i, neighbors in in_edges_r.items():
        for j in neighbors:
            num += beta(i, j)*(A_r.dot(Y[j]) - Y[i])
            denom += beta(i, j)

    for i, neighbors in neg_edges_r.items():
        for j in neighbors:
            num -= beta(i, j)*(A_r.dot(Y[j]) - Y[i])
            denom -= beta(i, j)

    return num / denom


def calc_B(Y, A, beta, in_edges, neg_edges):
    """ Calculates new bias vectors for each edge type according to Eq 4.

    Parameters
    ----------
    Y         : np.array, entity embeddings
    A         : dict that maps edge type to np.array
    beta      : func from in_edges[0].keys() to float
    in_edges  : dict of dict of incoming edges
    neg_edges : dict of dict of incoming non-edges

    Returns
    -------
    dict that maps edge type to np.array
    """
    return {r: calc_Br(Y, A[r], lambda i, j: beta(i, j, r), in_edges[r], neg_edges[r])
            for r in in_edges.keys()}


def calc_loss(X, Y, A, B, alpha, beta, lam, in_edges, neg_edges):
    """ Calculates a loss of the current model according to Eq 3.

    Parameters
    ----------
    X         : np.array, distributional embeddings
    Y         : np.array, current embeddings
    B         : dict that maps edge type to np.array
    alpha     : func from 'edges.keys()' to float
    beta      : func from 'edges.keys()' to float
    lam       : float regularization parameter
    in_edges  : dict that maps entity index to list of neighbors
    neg_edges : dict that maps entity index to list of non-neighbors

    Returns
    -------
    float
    """
    loss = 0.
    for r in in_edges.keys():
        for i, neighbors in in_edges[r].items():
            for j in neighbors:
                loss += beta(i, j, r)*np.linalg.norm(A[r].dot(Y[j]) + B[r] - Y[i], ord=2)
        for i, neighbors in neg_edges[r].items():
            for j in neighbors:
                loss -= beta(i, j, r)*np.linalg.norm(A[r].dot(Y[j]) + B[r] - Y[i], ord=2)

        loss += lam*np.linalg.norm(A[r], ord=2)

    for i in range(len(X)):
        loss += alpha(i)*np.linalg.norm(X[i]-Y[i], ord=2)

    return loss

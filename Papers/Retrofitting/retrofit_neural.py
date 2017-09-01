from sklearn.utils import shuffle
from itertools import product
import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import normalize
import time


def g_r(q_i, q_j, A_r, u_r, b_r):
    """ Helper function for the loss. 
    
    Parameters
    ----------
    q_i : np.array for the embedding of entity i.
    q_j : np.array for the embedding of entity j.
    A_r : np.array for the first layer weights.
    u_r : np.array for the second layer weights.
    b_r : np.array for bias vector.
    """
    a = np.tanh(A_r.dot(q_j) + b_r)
    b = q_i
    return u_r.dot(np.vstack((a, b)))


def f_r(q_i, q_j, A_r, u_r, b_r):
    """ Pairwise penalty function.

    Parameters
    ----------
    q_i : np.array for the embedding of entity i.
    q_j : np.array for the embedding of entity j.
    A_r : np.array for the first layer weights.
    U_r : np.array for the second layer weights.
    b_r : np.array for the bias vector.

    Returns
    -------
    Scalar of the evaluation of the penalty function.
    """
    return g_r(q_i, q_j, A_r, u_r, b_r)**2


def grad_Y(X, Y, A, u, b, k, lam, batch, alpha, beta, in_edges, out_edges,
           neg_in_edges, neg_out_edges):
    """ Calculates the partial derivative with respect to Y.

    Parameters
    ----------
    X            : np.array (distributional embeddings)
    Y            : np.array, current embeddings
    A            : dict that maps edge type to np.array, first layer weights
    u            : dict that maps edge type to np.array, second layer weights
    b            : dict that maps edge type to np.array, bias
    k            : hidden unit size
    lam          : float L2 regularization coefficient
    batch        : list of indices to optimize for
    alpha        : func from `edges[i].keys()` to floats or None
    beta         : func from `edges[i].keys()` to floats or None
    in_edges     : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an incoming edge
    out_edges    : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an outgoing edge
    neg_in_edges : dict that maps edge type to dict that maps entity index
                   to list of non-neighbors by an incoming edge
    neg_out_edges : dict that maps edge type to dict that maps entity index
                    to list of non-neighbors by an outgoing edge

    Returns
    -------
    dY : np.array
    """

    dY = np.zeros_like(Y)
    n_nodes = len(Y)
    for i in batch:
        dY[i] = alpha(i)*(Y[i] - X[i])
        for r in in_edges.keys():   # Y[i] functions as q_i in these edges.
            for j in in_edges[r][i]:
                dY[i] += beta(i, j, r)*(u[r][:, k:]).T.dot(g_r(Y[i], Y[j], A[r], u[r], b[r]))
            for j in neg_in_edges[r][i]:
                dY[i] -= beta(i, j, r)*(u[r][:, k:]).T.dot(g_r(Y[i], Y[j], A[r], u[r], b[r]))
        for r in out_edges.keys():  # Y[i] functions as q_j in these edges.
            for j in out_edges[r][i]:
                q_i = Y[j]
                q_j = Y[i]

                x1 = A[r].T.dot(u[r][:, :k].T.dot(g_r(q_i, q_j, A[r], u[r], b[r])))
                x2 = np.tanh(A[r].dot(q_j) + b[r]).T
                x3 = (1 - x2**2).T

                dY[i] += beta(j, i, r)*x1.dot(x2).dot(x3)
            for j in neg_out_edges[r][i]:
                q_i = Y[j]
                q_j = Y[i]

                x1 = A[r].T.dot(u[r][:, :k].T.dot(g_r(q_i, q_j, A[r], u[r], b[r])))
                x2 = np.tanh(A[r].dot(q_j) + b[r]).T
                x3 = (1 - x2**2).T

                dY[i] -= beta(j, i, r)*x1.dot(x2).dot(x3)
    return dY


def grad_b_helper(q_i, q_j, A_r, u_r, b_r, k):
    """ Helper function for calculating the partial wrt b.

    Parameters
    ----------
    q_i : Embedding for entity i
    q_j : Embedding for entity j
    A_r : np.array
    u_r : np.array
    b_r : np.array
    k   : int

    Returns
    -------
    np.array    
    """
    x1 = (u_r[:, :k]).T.dot(g_r(q_i, q_j, A_r, u_r, b_r))
    x2 = np.tanh(A_r.dot(q_j) + b_r).T
    x3 = (1 - x2**2).T
    return x1.dot(x2.dot(x3))


def grad_B(X, Y, A, u, b, k, lam, batch, alpha, beta, in_edges, out_edges,
           neg_in_edges, neg_out_edges):
    """ Calculates the partial derivative with respect to B.

    Parameters
    ----------
    X            : np.array (distributional embeddings)
    Y            : np.array, current embeddings
    A            : dict that maps edge type to np.array, first layer weights
    u            : dict that maps edge type to np.array, second layer weights
    b            : dict that maps edge type to np.array, bias
    k            : hidden unit size
    lam          : float L2 regularization coefficient
    batch        : list of indices to optimize for
    alpha        : func from `edges[i].keys()` to floats or None
    beta         : func from `edges[i].keys()` to floats or None
    in_edges     : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an incoming edge
    out_edges    : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an outgoing edge
    neg_in_edges : dict that maps edge type to dict that maps entity index
                   to list of non-neighbors by an incoming edge
    neg_out_edges : dict that maps edge type to dict that maps entity index
                    to list of non-neighbors by an outgoing edge

    Returns
    -------
    dB : np.array
    """
    dB = {r: lam*b[r] for r in b.keys()}
    for r in in_edges.keys():
        for i in batch:
            for j in in_edges[r][i]:
                dB[r] += beta(i, j, r)*grad_b_helper(Y[i], Y[j], A[r], u[r], b[r], k)
    for r in neg_in_edges.keys():
        for i in batch:
            for j in neg_in_edges[r][i]:
                dB[r] -= beta(i, j, r)*grad_b_helper(Y[i], Y[j], A[r], u[r], b[r], k)
    return dB


def grad_a_helper(q_i, q_j, A_r, u_r, b_r, k):
    """ Helper function for calculating the partial wrt A_r.

    Parameters
    ----------
    q_i : Embedding for entity i
    q_j : Embedding for entity j
    A_r : np.array
    u_r : np.array
    b_r : np.array
    k   : int

    Returns
    -------
    np.array    
    """
    return grad_b_helper(q_i, q_j, A_r, u_r, b_r, k).dot(q_j.T)


def grad_A(X, Y, A, u, b, k, lam, batch, alpha, beta, in_edges, out_edges,
           neg_in_edges, neg_out_edges):
    """ Calculates the partial derivative with respect to A.

    Parameters
    ----------
    X            : np.array (distributional embeddings)
    Y            : np.array, current embeddings
    A            : dict that maps edge type to np.array, first layer weights
    u            : dict that maps edge type to np.array, second layer weights
    b            : dict that maps edge type to np.array, bias
    k            : hidden unit size
    lam          : float L2 regularization coefficient
    batch        : list of indices to optimize for
    alpha        : func from `edges[i].keys()` to floats or None
    beta         : func from `edges[i].keys()` to floats or None
    in_edges     : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an incoming edge
    out_edges    : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an outgoing edge
    neg_in_edges : dict that maps edge type to dict that maps entity index
                   to list of non-neighbors by an incoming edge
    neg_out_edges : dict that maps edge type to dict that maps entity index
                    to list of non-neighbors by an outgoing edge

    Returns
    -------
    dA : np.array
    """
    dA = {r: lam*A[r] for r in A.keys()}
    for r in in_edges.keys():
        for i in batch:
            for j in in_edges[r][i]:
                dA[r] += beta(i, j, r)*grad_a_helper(Y[i], Y[j], A[r], u[r], b[r], k)
    for r in neg_in_edges.keys():
        for i in batch:
            for j in neg_in_edges[r][i]:
                dA[r] -= beta(i, j, r)*grad_a_helper(Y[i], Y[j], A[r], u[r], b[r], k)
    return dA


def grad_u_helper(q_i, q_j, A_r, u_r, b_r, k):
    """ Helper function for calculating the partial wrt u_r.

    Parameters
    ----------
    q_i : Embedding for entity i
    q_j : Embedding for entity j
    A_r : np.array
    u_r : np.array
    b_r : np.array
    k   : int

    Returns
    -------
    np.array    
    """
    return g_r(q_i, q_j, A_r, u_r, b_r).dot(np.vstack((np.tanh(A_r.dot(q_j) + b_r), q_i)).T)


def grad_U(X, Y, A, u, b, k, lam, batch, alpha, beta, in_edges, out_edges,
           neg_in_edges, neg_out_edges):
    """ Calculates the partial derivative with respect to u_r.

    Parameters
    ----------
    X            : np.array (distributional embeddings)
    Y            : np.array, current embeddings
    A            : dict that maps edge type to np.array, first layer weights
    u            : dict that maps edge type to np.array, second layer weights
    b            : dict that maps edge type to np.array, bias
    k            : hidden unit size
    lam          : float L2 regularization coefficient
    batch        : list of indices to optimize for
    alpha        : func from `edges[i].keys()` to floats or None
    beta         : func from `edges[i].keys()` to floats or None
    in_edges     : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an incoming edge
    out_edges    : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an outgoing edge
    neg_in_edges : dict that maps edge type to dict that maps entity index
                   to list of non-neighbors by an incoming edge
    neg_out_edges : dict that maps edge type to dict that maps entity index
                    to list of non-neighbors by an outgoing edge

    Returns
    -------
    dA : np.array
    """
    dU = {r: lam*u[r] for r in u.keys()}
    for r in in_edges.keys():
        for i in batch:
            for j in in_edges[r][i]:
                dU[r] += beta(i, j, r)*grad_u_helper(Y[i], Y[j], A[r], u[r], b[r], k)
            for j in neg_in_edges[r][i]:
                dU[r] -= beta(i, j, r)*grad_u_helper(Y[i], Y[j], A[r], u[r], b[r], k)
    return dU


def calc_loss_neural(X, Y, A, u, b, k, lam, batch, alpha, beta, in_edges, out_edges,
                     neg_in_edges, neg_out_edges):
    """ Calculates the loss on a validation batch.

    Parameters
    ----------
    X            : np.array (distributional embeddings)
    Y            : np.array, current embeddings
    A            : dict that maps edge type to np.array, first layer weights
    u            : dict that maps edge type to np.array, second layer weights
    b            : dict that maps edge type to np.array, bias
    k            : hidden unit size
    lam          : float L2 regularization coefficient
    batch        : list of indices to optimize for
    alpha        : func from `edges[i].keys()` to floats or None
    beta         : func from `edges[i].keys()` to floats or None
    in_edges     : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an incoming edge
    out_edges    : dict that maps edge type to dict that maps entity index
                   to list of neighbors by an outgoing edge
    neg_in_edges : dict that maps edge type to dict that maps entity index
                   to list of non-neighbors by an incoming edge
    neg_out_edges : dict that maps edge type to dict that maps entity index
                    to list of non-neighbors by an outgoing edge

    Returns
    -------
    dA : np.array
    """
    loss = lam*(sum([np.linalg.norm(A_r, ord=2) for A_r in A.values()])
                + sum([np.linalg.norm(u_r, ord=2) for u_r in u.values()])
                + sum([np.linalg.norm(b_r, ord=2) for b_r in b.values()])
                )

    for i in batch:
        loss += alpha(i)*np.linalg.norm(Y[i] - X[i], ord=2)

    for r, edges_r in in_edges.items():
        for i in batch:
            for j in edges_r[i]:
                loss += beta(i, j, r)*f_r(Y[i], Y[j], A[r], u[r], b[r])
            for j in neg_in_edges[r][i]:
                loss -= beta(i, j, r)*f_r(Y[i], Y[j], A[r], u[r], b[r])

    return np.asscalar(loss)


def retrofit_neural(X, in_edges, out_edges, k=5, n_iter=100, alpha=None,
                    beta=None, tol=1e-2, lr=0.5, lam=1e-5, verbose=0,
                    lr_decay=0.9, batch_size=32, patience=20):
    """ Retrofit according to the neural penalty function.

    Parameters
    ----------
    X          : np.array (distributional embeddings)
    in_edges   : dict that maps edge type to dict that maps entity index
                 to list of neighbors by an incoming edge
    out_edges  : dict that maps edge type to dict that maps entity index
                 to list of neighbors by an outgoing edge
    k          : hidden unit size
    n_iter     : int indicating the maximum number of iterations
    alpha      : func from `edges[i].keys()` to floats or None
    beta       : func from `edges[i].keys()` to floats or None
    tol        : float
                 If the average distance change between two rounds is at or
                 below this value, we stop. Default to 10^-2 as suggested
                 in the Faruqui et al paper.
    lr         : float learning rate
    lam        : float L2 regularization coefficient
    verbose    : int indicating how often to print intermediate results. 0 never prints.
    lr_decay   : float learning rate decay
    batch_size : int size of the SGD batch
    patience   : int number of iterations with increasing loss to permit before stopping.
    
    Returns
    -------
    Y : np.array, same dimensions and arrangement as `X`.
    A : dict that maps edge_type to an np.array
    U : dict that maps edge_type to an np.array
    B : dict that maps edge_type to an np.array
    """
    n_relation_types = len(in_edges)
    n_nodes = len(X)
    if not alpha:
        alpha = lambda i: 1
    if not beta:
        beta = lambda i, j, r: 1 / max(
            [np.sum([len(er[i]) for er in in_edges.values()]), 1])*(
            int(j in in_edges[r][i])+0.1)

    X = np.expand_dims(X, axis=2)
    Y = X.copy()
    Y_prev = Y.copy()
    A_prev = {}
    U_prev = {}
    B_prev = {}

    for rel, edges_r in in_edges.items():
        for i, neighbors in in_edges[rel].items():
            if len(neighbors) > 0:
                j = neighbors[0]
                d1 = Y[i].shape[0]
                d2 = Y[j].shape[0]
                if k == d1 and d1 == d2:
                    A_prev[rel] = np.eye(k)
                else:
                    A_prev[rel] = np.random.normal(0, 1, size=(k, d2))
                U_prev[rel] = np.hstack((np.ones((1, k)), -np.ones((1, d1))))
                B_prev[rel] = np.zeros((k, 1))
                break
            if i == len(neighbors) - 1:
                print("A[{}] has 0 edges.".format(r))

    A = A_prev.copy()
    U = U_prev.copy()
    B = B_prev.copy()

    prev_loss = np.inf
    for iteration in range(1, n_iter+1):
        if verbose:
            print("Iteration {} of {}".format(iteration, n_iter), end='\r')

        batch = np.random.choice(n_nodes, size=batch_size)

        neg_in_edges = {r: {i: [] for i in range(n_nodes)} for r in in_edges.keys()}
        neg_out_edges = {r: {i: [] for i in range(n_nodes)} for r in in_edges.keys()}
        for r, in_edges_r in in_edges.items():
            for i in batch:
                neg_in_edges[r][i] = np.random.choice(n_nodes, size=len(in_edges_r[i]))
                for j in neg_in_edges[r][i]:
                    neg_out_edges[r][j].append(i)
        #print("Calculating dB...", end='\r')
        dB = grad_B(X, Y, A, U, B, k, lam, batch, alpha, beta, in_edges, out_edges,
                    neg_in_edges, neg_out_edges)
        B = {r: B_prev[r] - lr*np.clip(dB[r], -1e3, 1e3) for r in in_edges.keys()}
        if np.any([np.any(np.isnan(U[r])) for r in in_edges.keys()]):
            print("B Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev, A_prev, U_prev, B_prev)

        #print("Calculating dU...", end='\r')
        dU = grad_U(X, Y, A, U, B, k, lam, batch, alpha, beta, in_edges, out_edges,
                    neg_in_edges, neg_out_edges)
        U = {r: U_prev[r] - lr*np.clip(dU[r], -1e3, 1e3) for r in in_edges.keys()}
        if np.any([np.any(np.isnan(U[r])) for r in in_edges.keys()]):
            print("U Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev), A_prev, U_prev, B_prev

        #print("Calculating dA...", end='\r')
        dA = grad_A(X, Y, A, U, B, k, lam, batch, alpha, beta, in_edges, out_edges,
                    neg_in_edges, neg_out_edges)
        A = {r: A_prev[r] - lr*np.clip(dA[r], -1e3, 1e3) for r in in_edges.keys()}
        if np.any([np.any(np.any(np.isnan(A[r]))) for r in in_edges.keys()]):
            print("A Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev), A_prev, U_prev, B_prev

        #print("Calculating dY...", end='\r')
        dY = grad_Y(X, Y, A, U, B, k, lam, batch, alpha, beta, in_edges, out_edges,
                    neg_in_edges, neg_out_edges)
        Y = Y - lr*np.clip(dY, -1e3, 1e3)
        if np.any(np.any(np.isnan(Y))):
            print("Y Diverged at iteration {}".format(iteration))
            return np.squeeze(Y_prev), A_prev, U_prev, B_prev

        val_batch = np.random.choice(n_nodes, size=batch_size)
        loss = calc_loss_neural(X, Y, A, U, B, k, lam, val_batch, alpha, beta,
                                in_edges, out_edges, neg_in_edges, neg_out_edges)
        if loss > prev_loss:
            patience -= 1
            if patience < 0:
                print("Loss reached local minimum (and patience expired) at iteration {}".format(iteration-1))
                return np.squeeze(Y_prev), A_prev, U_prev, B_prev
        prev_loss = loss

        changes = np.mean(np.abs(np.linalg.norm(np.squeeze(Y_prev[batch]) - np.squeeze(Y[batch]), ord=2)))
        if verbose and iteration % verbose == 0:
            print("Iteration {:d} of {:d}\tChanges: {:.5f}\tLoss: {:.3f}\tPatience: {:d}".format(iteration, n_iter, changes, loss, patience))
        if changes <= tol:
            if verbose:
                print("Converged at iteration {}".format(iteration))
            return np.squeeze(Y), A, U, B
        else:
            Y_prev = Y.copy()
            A_prev = A.copy()
            U_prev = U.copy()
            B_prev = B.copy()
            lr *= lr_decay
    if verbose:
        print("Stopping at iteration {:d}; change was {:.3f}".format(iteration, changes))
    return np.squeeze(Y), A, U, B

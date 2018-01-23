import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import Imputer


def save_obj(obj, name):
    """ Pickles obj into a file at "obj/${name}.pkl"

    Parameters
    ----------
    obj: The object to be saved.
    name: The filename to be used.
    """
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """ Loads from a file at "obj/${name}.pkl"

    Parameters
    ----------
    name: The filename to be loaded.

    Returns
    ----------
    The contents of the file.
    """
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plot_embeddings(X, title, entity_types, entity_counts, n_iter=200):
    """ Plots TSNE embeddings. Assumes that only 2 types of entities are present.

    Parameters
    ----------
    X : np.array of representations
    title : str plot title
    entity_types : list of str entity type names
    entity_counts : list of int entity counts
    n_iter : int maximum number of TSNE iterations to allow
    """
    t = TSNE(n_iter=n_iter)
    X_tsne = t.fit_transform(X)
    plt.scatter(X_tsne[:entity_counts[0], 0], X_tsne[:entity_counts[0], 1], color='r', marker='.')
    plt.scatter(X_tsne[entity_counts[0]:, 0], X_tsne[entity_counts[0]:, 1], color='b', marker='.')
    plt.legend(entity_types)
    plt.title(title)
    plt.xlabel("tSNE Component 1")
    plt.ylabel("tSNE Component 2")
    plt.show()


def get_train_test_indices(edges, p_train=0.7, neg_sampling=1.0):
    """ Selects train and test examples.

    Parameters
    ----------
    edges       : dict mapping from a node index to a list of neighbors.
    p_train     : float that indicates that probability of a sample being
                  assigned to the training set. All edges that originate
                  at the same node are assigned to the same set.
    neg_sampling: float that indicates the number of "negative" edges per
                  "positive" edge.

    Returns
    ----------
    train_indices     : list of (i, j) that occurs in edges.
    test_indices      : list of (i, j) that occurs in edges.
    neg_train_indices : list of (i, j) that does not occur in edges.
    neg_test_indices  : list of (i, j) that does not occur in edges.
    """
    train_indices = []
    test_indices = []
    neg_train_indices = []
    neg_test_indices = []
    all_nodes = set(range(len(edges.items())))
    for num, (i, neighbors) in enumerate(edges.items()):
        if num % 1000 == 0:
            print(num, end='\r')
        if len(neighbors) == 0:
            continue
        all_non_neighbors = list(all_nodes - set(neighbors))
        pos_indices = [(i, j) for j in neighbors]
        neg_indices = [(i, j) for j in np.random.choice(all_non_neighbors,
            size=int(len(neighbors)*neg_sampling), replace=True)]
        if np.random.uniform() < p_train:
            train_indices.extend(pos_indices)
            neg_train_indices.extend(neg_indices)
        else:
            test_indices.extend(pos_indices)
            neg_test_indices.extend(neg_indices)
    if len(train_indices) + len(neg_train_indices) == 0 or len(test_indices) + len(neg_test_indices)== 0:
        return get_train_test_indices(edges, p_train, neg_sampling)
    return train_indices, test_indices, neg_train_indices, neg_test_indices


def print_edge_counts(edges):
    """ Print the edge counts.

    Parameters
    ----------
    edges: dict that maps edge type to a dict that maps index to list of neighbors.
    """
    for edge_type, edge_dict in edges.items():
        n_edges = sum(len(neighbors) for neighbors in edge_dict.values())
        print("{:d} Edges of Type: {}".format(n_edges, edge_type))


def impute(M):
    """ Impute values to fill the empty entries of the matrix M.

    Parameters
    ---------
    M : np matrix to be imputed

    Returns
    ---------
    M_imputed : M with missing values imputed.
    """
    start = time.time()
    imputer = Imputer()
    M_imputed = imputer.fit_transform(M)
    print("Took {} seconds".format(time.time() - start))
    return M_imputed


def pmi_transform(M):
    """ Perform a PMI transform on M.

    Parameters
    ----------
    M : np matrix to be transformed. M must be square.

    Returns
    ----------
    PMI transform of M
    """
    assert(M.shape[0] == M.shape[1])
    pmi = PMITransformer()
    return pmi.fit_transform(M)


def reduce_dims(M, n_components, verbose=True):
    """ Reduce M to n_components dimensions.

    Parameters
    ----------
    M            : np matrix
    n_components : int number of components to retain.
    verbose      : bool to indicate whether to print the explained variance.

    Returns
    ---------
    M_small      : dim-reduced version of M.
    """
    svd = TruncatedSVD(n_components=n_components)
    M_small=svd.fit_transform(M)
    if verbose:
        print("Explained variance by SVD component: {}".format(svd.explained_variance_ratio_))
    return M_small


def vectorize(co_occurrences, min_threshold=0, max_threshold=np.inf,
              verbose=True):
    """ Vectorizes co_occurrence data.

    Parameters
    ----------
    co_occurrences : a dict that maps term to a dict that maps tokens to counts.
    min_threshold  : the minimum number of occurrences of a token to have it count.
    max_threshold  : the maximum number of occurrences of a token to have it count.
    verbose        : bool indicating whether to print reports.

    Returns
    -------
    M                 : np matrix containing the thresholded co-occurrence counts.
    row_names         : list of the row names in order.
    column_names      : list of the retained column names in order.
    row_names_dict    : dict mapping row name to row index.
    column_names_dict : dict mapping column name to column index.
    """
    start = time.time()

    C = []
    row_names = []
    for row_name, features in co_occurrences.items():
        row_names.append(row_name)
        C.append(features)
    row_names = np.array(row_names)

    # Vectorize
    v = DictVectorizer(sparse=True)
    M = v.fit_transform(C)
    column_names = np.array(v.get_feature_names())
    if verbose:
        print("Unthresholded shape: {}".format(M.shape))

    # Min Threshold
    row_names = row_names[M.getnnz(1) > min_threshold]
    column_names = column_names[M.getnnz(0) > min_threshold]
    M = M[M.getnnz(1) > min_threshold][:,M.getnnz(0) > min_threshold]
    if verbose:
        print("Min Thresholded shape: {}".format(M.shape))

    # Max Threshold
    column_names = column_names[M.getnnz(0) < max_threshold]
    M = M[: , M.getnnz(0) < max_threshold]
    if verbose:
        print("Max Thresholded shape: {}".format(M.shape))

    row_names_dict = {row_name: i for i, row_name in enumerate(row_names)}
    column_names_dict = {name: i for i, name in enumerate(column_names)}
    if verbose:
        print("Took {} seconds".format(time.time() - start))
    return M, row_names, column_names, row_names_dict, column_names_dict

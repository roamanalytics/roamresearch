from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


def print_top_n_not_annotated(n, probs, Y_test, index_by_name,
                              test_indices):
    """ Print the top n predicted edges that are not annotated.

    Parameters
    ----------
    n             : int, number of predicted edges to print.
    probs         : np.array, predicted probabilities of each sample.
    Y_test        : np.array, the true labels.
    index_by_name : dict, maps row names to indices
    test_indices  : list of indices used in the test set
    """
    probs = probs[:, 1]
    i = 0
    name_by_index = {}
    for name, index in index_by_name.items():
        name_by_index[index] = name

    probs_sorted = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for (index, prob) in probs_sorted:
        if i >= n:
            break
        if Y_test[index] == 0:
            i += 1
            print("Predicted drug '{}' TREATS '{}' with score: {:.4f}.".format(
                    name_by_index[test_indices[index][0]], 
                    name_by_index[test_indices[index][1]],
                    prob))


def print_predictions(model, X_test, Y_test, index_by_name, test_indices, n=15):
    """ Print the confidently predicted edges that are not annotated.

    Parameters
    ----------
    model         : trained prediction model
    X_test        : testing set
    Y_test        : testing labels
    index_by_name : dict, maps row names to indices
    test_indices  : list of indices used in the test set
    n             : int number of predicted edges to print
    """
    probs = model.predict_proba(X_test)
    print_top_n_not_annotated(n, probs, Y_test, index_by_name, test_indices)
            

def evaluate(X, Y_train, Y_test, make_train, make_test, clf):
    """ Evaluate representations by link prediction accuracy.

    Parameters
    ----------
    X          : np.array of representations
    Y_train    : np.array of training labels
    Y_test     : np.array of testing labels
    make_train : function that returns training set from X
    make_test  : function that returns testing set from X
    clf        : classifier to train and test
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_train = make_train(X_normalized)
    X_test = make_test(X_normalized)
    clf.fit(X_train, Y_train)
    return clf.score(X_test, Y_test), clf, X_train, X_test
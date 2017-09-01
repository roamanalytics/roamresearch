from collections import defaultdict, Counter
import csv
import os
import scipy
vsmdata_home = "Evaluation/vsmdata"

# Evaluation code. Adapted from Chris Potts's [CS244u course notes]
# (http://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/vsm.ipynb#In-class-bake-off:-Word-similarity).

def wordsim_dataset_reader(src_filename, header=False, delimiter=','):    
    """Basic reader that works for all four files, since they all have the 
    format word1,word2,score, differing only in whether or not they include 
    a header line and what delimiter they use.
    
    Parameters
    ----------
    src_filename : str
        Full path to the source file.
        
    header : bool (default: False)
        Whether `src_filename` has a header.
        
    delimiter : str (default: ',')
        Field delimiter in `src_filename`.
    
    Yields
    ------    
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity 
       score in the file so that we are intuitively aligned with our 
       distance-based code.
    
    """
    reader = csv.reader(open(src_filename), delimiter=delimiter)
    if header:
        next(reader)
    for row in reader:
        w1, w2, score = row
        # Negative of scores to align intuitively with distance functions:
        score = -float(score)
        yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'wordsim353.csv')
    return wordsim_dataset_reader(src_filename, header=True)
 

def mturk287_reader():
    """MTurk-287: http://tx.technion.ac.il/~kirar/Datasets.html"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'MTurk-287.csv')
    return wordsim_dataset_reader(src_filename, header=False)

    
def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'MTURK-771.csv')
    return wordsim_dataset_reader(src_filename, header=False)


def cosine(u, v):        
    """Cosine distance between 1d np.arrays `u` and `v`, which must have 
    the same dimensionality. Returns a float."""
    return scipy.spatial.distance.cosine(u, v)


def word_similarity_evaluation(reader, mat, row_ids, distfunc=cosine):
    """Word-similarity evalution framework.
    
    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).
    
    mat : 2d np.array
        The VSM being evaluated.
        
    rownames : dict
        The names of the rows in mat.
        
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`, 
        `matching`, `jaccard`, as well as any other distance measure 
        between 1d vectors.  
    
    Prints
    ------
    To standard output
        Size of the vocabulary overlap between the evaluation set and
        rownames. We limit the evalation to the overlap, paying no price
        for missing words (which is not fair, but it's reasonable given
        that we're working with very small VSMs in this notebook).
    
    Returns
    -------
    float
        The Spearman rank correlation coefficient between the dataset
        scores and the similarity values obtained from `mat` using 
        `distfunc`. This evaluation is sensitive only to rankings, not
        to absolute values.
    
    """    
    sims = defaultdict(list)
    vocab = set([])
    for w1, w2, score in reader():
        if w1 in row_ids and w2 in row_ids:
            sims[w1].append((w2, score))
            sims[w2].append((w1, score))
            vocab.add(w1)
            vocab.add(w2)
    print("Evaluation vocabulary size: %s" % len(vocab))
    # Evaluate the matrix by creating a vector of all_scores for data
    # and all_dists for mat's distances. 
    all_scores = []
    all_dists = []
    for word in vocab:
        vec = mat[row_ids[word]]
        vals = sims[word]
        cmps, scores = zip(*vals)
        all_scores += scores
        all_dists += [distfunc(vec, mat[row_ids[w]]) for w in cmps]
    # Return just the rank correlation coefficient (index [1] would be the p-value):
    return scipy.stats.spearmanr(all_scores, all_dists)[0]


def full_word_similarity_evaluation(mat, row_ids):
    """Evaluate the (mat, rownames) VSM against all four datasets."""
    for reader in (wordsim353_reader, mturk771_reader, mturk287_reader):
        print("-"*40)
        print(reader.__name__)
        print('Spearman r: %0.03f' % word_similarity_evaluation(reader, mat, row_ids))
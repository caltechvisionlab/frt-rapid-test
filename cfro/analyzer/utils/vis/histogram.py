import numpy as np

SMALL_NUM = 1.0e-10  # used in the histogram function to avoid runding erros


def my_histogram(X, normalization=True):
    '''Compute histogram of points.
    Estimate how many bins it is reasonable to have.
    Return the centers of the bins rather than the edges of the bins.
    Normalize in order to approximate a probability density function. '''

    X = X.flatten()  # if X is a matrix transform it into a vector
    N = X.size
    X_MAX = X.max()
    X_MIN = X.min()
    N_BINS = np.ceil(
        np.sqrt(N))  # bins should not be too many, and should not grow too fast with N, hence the square root
    BIN_STEP_SIZE = (X_MAX - X_MIN) / N_BINS
    # print(f'X_MIN,X_MAX = ({X_MIN:0.3},{X_MAX:0.3}), n. bins = {N_BINS}, step size={BIN_STEP_SIZE:0.3}')

    BIN_EDGES = np.arange(X_MIN, X_MAX + BIN_STEP_SIZE + SMALL_NUM, BIN_STEP_SIZE)
    BIN_CENTERS = np.arange(X_MIN + BIN_STEP_SIZE / 2, X_MAX + BIN_STEP_SIZE / 2 + SMALL_NUM, BIN_STEP_SIZE)

    counts, bin_edges = np.histogram(X, bins=BIN_EDGES)

    if normalization:
        NORMALIZATION_FACTOR = N * BIN_STEP_SIZE
        return counts / NORMALIZATION_FACTOR, BIN_CENTERS
    else:
        return counts, BIN_CENTERS

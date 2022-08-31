import numpy as np
import matplotlib.pyplot as plt

from cfro.analyzer.utils.vis.histogram import my_histogram

SMALL_NUM = 1.0e-10  # used in the histogram function to avoid rounding errors


def FNMR_vs_FMR(C, Y, error_range=(0.0, 1.0)):
    '''Computes the false non match rate and false match rate as a fuction of a
    threshold apply to the confidence value C. The inputs are two arrays:
    C = list of confidence values that are assigned to data pairs
    Y = ground truth label, where 0 means non-match and 1 means match
    The function returns three arrays:
    FNMR - false non match rate at a given threshold
    FMR - false match rate at a given threshold
    thres - thresholds at which the FNMR and FMR values are computed'''

    # just in case C and Y are matrices
    C = C.flatten()
    Y = Y.flatten()

    # check a couple of things before proceeding
    if len(Y) != len(C):
        print('FNMR_vs_FMR: the length of Y and C should be the same.')
        return

    if set(Y) != {0, 1}:
        print('FNMR_vs_FMR: Y should only contain values 0 and 1. It contains {set(Y)}')
        return

    N = len(C)  # number of items

    # sort the entries in C and Y so that the C are in non-decreasing order
    indices = np.argsort(C)
    C = C[indices]
    Y = Y[indices]

    # place the thresholds in between the C values
    # The value of thresh[i] will be equal or greater than C[i]
    thresh = (C[:-1] + C[1:]) / 2

    Y1_cumsum = np.cumsum(Y)  # counts how many positives are below or equal to a given threshold
    Y0_cumsum = np.cumsum(1 - Y)  # counts how many negatives are below or equal to a given threshold
    N1 = Y1_cumsum[-1]  # how many items are of type 1
    N0 = N - N1  # how many items are of type 0

    FMR = 1. - (Y0_cumsum / N0)  # fraction of negatives whose confidence falls above threshold
    FNMR = Y1_cumsum / N1  # fraction of positives whose confidence falls below or equal to threshold

    lower_bound = C[0] - 1  # Assuming C[0] is the smallest value, subtract 1 or choose an appropriate lower bound
    upper_bound = C[-1] + 1  # Assuming C[-1] is the largest value, add 1 or choose an appropriate upper bound
    thresh = np.hstack(([lower_bound], (C[:-1] + C[1:]) / 2, [upper_bound]))

    i_of_interest = np.where(
        (FNMR >= error_range[0]) & (FMR >= error_range[0]) & (FNMR <= error_range[1]) & (FMR <= error_range[1]))[0]

    if len(i_of_interest) > 0:
        return FNMR[i_of_interest], FMR[i_of_interest], thresh[i_of_interest]
    else:
        print(f"Could not create plots: FMR_FNMR ERROR_RANGE={error_range} is too narrow. Change and run again.")
        exit()


def test_FNMR_vs_FMR(N=100, means=[0, 2]):
    '''Just to test the above function'''

    # generate synthetic C and Y data
    C = np.empty(2 * N)
    Y = np.empty(2 * N)
    for i in range(len(means)):
        C[i * N:(i + 1) * N] = np.random.normal(means[i], 1., N)
        Y[i * N:(i + 1) * N] = i

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # plot the synthetic data
    for i in range(len(means)):
        c, b = my_histogram(C[np.where(Y == i)])
        ax[0].plot(b, c, label=i)
    ax[0].legend()

    # compute FNMR and FMR
    FNMR, FMR, thresh = FNMR_vs_FMR(C, Y)

    # plot individual FNMR and FMR curves vs the threshold
    ax[1].plot(thresh, FMR, label='FMR')
    ax[1].plot(thresh, FNMR, label='FNMR')
    ax[1].set_xlabel('thresh')
    ax[1].set_ylabel('error')
    ax[1].legend()

    # plot the FNMR vs FMR
    if len(FMR) > 0:
        ax[2].plot(FMR, FNMR)
        ax[2].set_xlabel('FMR')
        ax[2].set_ylabel('FNMR')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')

    plt.show()


if __name__ == '__main__':
    test_FNMR_vs_FMR()

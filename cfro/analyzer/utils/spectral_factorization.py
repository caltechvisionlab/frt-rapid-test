import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def orthonormal(A):
    '''test that the columns of A are orthonormal'''
    EPS = 10e-10
    B = A.transpose() @ A                    # orthonormal cols => then this matrix is the identity
    D, _ = B.shape                     # figure out the dimensions of the square matrix
    error = np.linalg.norm(B-np.identity(D))/D # this should be close to zero
    if error < EPS:
        return True
    else:
        print(f'orthonormal: test failed - error is {error}>{EPS}')
        return False


def matrix_loss(U):
    '''Figure of merit for the choice of U.
    Ideally the entries of U are nonnegative.
    The loss is the sum of the squares of the negative entries.'''
    ij = np.where(U < 0)  # find where the entries of U are negative
    neg_entries = U[ij]
    return np.sum(np.square(neg_entries))  # the loss is the sum of the squares of the negative entries


def unscramble_vectors(U, verbose=False):
    '''Rotate the columns of U so that they are (close to) nonnegative.
    Here the rotation matrix is represented as the exponential of a skew-symmetric
    matrix. This is practical because we do not have to deal with sines and cosines.'''

    D, K = U.shape  # D = n. of dimensions of space, K = n. of columns of U
    DX = 0.1  # initial magnitude of rotation in radiants
    MIN_DX = 1e-6  # if DX dips below this value it's pointless to continue, stop computation
    N_ITER_MAX = 1000  # maximum number of iterations
    TARGET_LOSS = 1e-2  # when loss is below this, then stop
    Ur = np.copy(U)  # copy of the scrambled vectors - eventually the "recovered U"
    S0 = np.zeros((K, K))  # basic skew-symmetric matrix -- exp(S0) is the identity
    loss = []  # list where we store the loss - for plotting if needed

    if K > D:
        print(f'unscramble_vectors: ERROR, expecting n_rows > n_cols, instead found n_rows={D} < n_cols={K}')

    # gradient descent on loss
    for i in range(N_ITER_MAX):

        # each (h,k) d.o.f. of the skew-symmetric matrix is a rotation axis
        for h in range(1, K):
            for k in range(0, h):

                # build a rotation matrix around the chosen rotation axis (h,k)
                S1 = np.copy(S0)
                S1[h, k] = DX
                S1[k, h] = -DX

                # compute rotation of U in two opposite directions
                Ua = U @ expm(S1)
                Ub = U @ expm(-S1)
                if orthonormal(Ub) == False:
                    print(f'Ub is not orthonormal')  # safety check in case numerics go bad

                # compute the loss (the sum of the squares of the negative entries in U)
                na = matrix_loss(Ua)
                nb = matrix_loss(Ub)

                # update the Ur matrix
                if na < nb:
                    U = Ua
                else:
                    U = Ub

        # take a note of the loss - if the loss is small enough then stop iterating
        loss.append(matrix_loss(U))
        if loss[-1] < TARGET_LOSS:
            break

        # if the loss increases, then decrease the step size
        # if the loss decreases then increase the step size
        if i > 0:
            if loss[i] > loss[i - 1]:
                DX = DX / 2
            else:
                DX = 1.1 * DX

        # if the step size is so small that it's pointless to continue, then stop
        if DX < MIN_DX:
            break

    if verbose:
        # if the desired loss was not achieved print a warning
        if len(loss) == N_ITER_MAX or DX < MIN_DX:
            print(f'Warning: unscramble_vectors reached {len(loss)} iterations, loss is {loss[-1]:0.3} and DX={DX:0.3}')
            fig = plt.figure(figsize=(14, 10))
            plt.plot(loss)
            plt.title('Unscramble loss', fontsize=24)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.xscale('log')
            plt.yscale('log')
        else:
            print(f'unscramble_vectors reached {len(loss)} iterations and loss is {loss[-1]:0.3}')

    return U, loss

class SpectralFactorizationClustering:

    def __init__(self, confidence_matrix: np.ndarray, thres_ratio=1.5, thres_magnitude=4.0, verbose=False):
        self.C = confidence_matrix
        self.N_tot = len(self.C)
        self.thres_ratio = thres_ratio
        self.thres_magnitude = thres_magnitude
        self.verbose = verbose

    def compute_number_of_clusters(self):
        '''Take a look at the singular values and decide how many clusters there are'''

        if self.verbose:
            print(f'compute_number_of_clusters: S = {self.S[0:5]}')

        # work backwards from the last singular vector and stop when sufficient
        #  evidence for a gap in magnitude is found
        K = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(self.N_tot - 1, 0, -1):
                if (self.S[i - 1] / self.S[i] > self.thres_ratio) & (self.S[i - 1] > self.thres_magnitude):
                    K = i
                    break
        return K  # the index i-1 correspond to the first viable singular value, but the numeration starts from zero


    def cluster(self, grouping_thres=0.2, method="regular", return_evs=False):
        ''' Use the SVD to compute the rank 1 approximation of a square matrix'''
        assert method in ["regular", "rotation"]

        # regular SVD
        U, S, VT = np.linalg.svd(self.C)  # Compute the SVD of the matrix A
        self.U = U
        self.S = S
        self.W = U * U * np.sign(U) * S[np.newaxis, :]  # this is the unnormalized version of U. Wij~1 when pt i belongs to cluster j
        self.r = np.zeros(len(S))  # we will store a diagnostic in this vector

        self.K_est = self.compute_number_of_clusters()
        if self.verbose:
            print(f'Estimating {self.K_est} clusters')

        if self.K_est == 1:
            self.Un = (self.U[:, 0] * np.sign(np.mean(U[:, 0]))).reshape(-1, 1)  # flip the sign if needed, Reshape to make 1D vector into 1D matrix
            self.Wn = self.W[:, 0].reshape(-1, 1)
            self.Sn = self.S[[0]]
        else:  # multiple vectors found - unscramble them whether this is needed or not
            # unscramble the vectors to make them as positive as possible
            # this may alter the order of the vectors
            Un, Un_loss = unscramble_vectors(U[:, 0:self.K_est], verbose=self.verbose)
            self.Un = Un
            S_est = []
            for k in range(self.K_est):
                S_est.append(Un[:, k].transpose() @ self.C @ Un[:, k])
            self.Sn = np.array(S_est)
            self.Wn = self.Un * self.Un * np.sign(self.Un) * self.Sn[np.newaxis, :]  # compute the non-normalized Un
            if self.verbose:
                print(f'Estimated S is {self.Sn}')

        # flip the sign of the U columns if needed to render them positive - this may hurt orthonormality
        # but only for vectors that are not useful
        for k in range(len(S)):
            self.U[:, k] *= np.sign(np.mean(self.U[:, k]))  # flip sign to make as positive as possible
            self.W[:, k] *= np.sign(np.mean(self.W[:, k]))  # flip sign to make as positive as possible

            # compute a diagnostic term that tells us if there are many negative entries
            non_positive_error = np.linalg.norm(U[:, k] - np.abs(U[:, k]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.r[k] = non_positive_error / np.linalg.norm(np.abs(
                    U[:, k]))  # if the entries are mostly one-signed then this should be close to 1, otherwise close to 0

        cluster_indexes = np.full_like(S, -1, dtype=int)
        _U = self.Un if method == "rotation" else self.U
        for k in range(self.K_est):  # assuming that the last cluster is just noise
            u = self.S[k] * _U[:, k] * _U[:, k] * np.sign(_U[:, k])
            ii = np.where(u > grouping_thres)[0]
            cluster_indexes[ii] = k
        if return_evs:
            return cluster_indexes, self.Wn
        return cluster_indexes
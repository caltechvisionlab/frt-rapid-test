from typing import Union

import numpy as np
import pandas as pd

def calculate_confidence_matrix(matches: pd.DataFrame,
                                bbox0_col: str = "bbox0_face_id",
                                bbox1_col: str = "bbox1_face_id",
                                conf_col: str = "confidence",
                                return_full_matrix=True) -> pd.DataFrame:
    # build a matrix with the confidence values for all pairs of images
    mdf = matches[[bbox0_col, bbox1_col, conf_col]]
    mdf = mdf.pivot(index=bbox0_col, columns=bbox1_col, values=conf_col)
    all_ids = sorted(list(set(mdf.columns) | set(mdf.index)), key=lambda x: [int(num) for num in x.split('-')])
    mdf = mdf.reindex(all_ids, axis=0).reindex(all_ids, axis=1)
    mdf[np.isnan(mdf)] = mdf.T  # fill in the missing values with the transposed values (symmetric matrix)

    # np.fill_diagonal(mdf.values, 1.0)  # fill in the diagonal with 1.0  TODO why is this failing
    for face_id in mdf.index:  # fill diagonal
        mdf.loc[face_id, face_id] = 1.0

    if return_full_matrix:
        mdf = mdf[~mdf.isna().any()].T[~mdf.T.isna().any()].T  # remove rows and columns that are all NaN

    return mdf

def calculate_rank1_approximation(C: Union[pd.DataFrame, np.ndarray], return_eigenvector_ratio=False):
    num_face_ids = len(C)
    # compute rank 1 approximation of matrix
    if num_face_ids > 0:  # if the matrix is empty this is pointless
        u, s, v = np.linalg.svd(C, full_matrices=True)
        u0 = np.sqrt(s[0]) * u[:, 0].reshape(num_face_ids, 1)
        if np.median(u0) < 0:  # the first eigenvector ought to be all positive or all negative
            u0 = -u0  # if negative, flip it
        C1 = u0 @ u0.T  # compute rank 1 approximation of C

        ev_ratio = s[0] / s[1]

    else:  # this is when the matrix is empty
        C1 = C
        u0 = np.array([])
        ev_ratio = None

    C1 = C1.clip(0.0, 1.0)  # clip values to be between 0 and 1

    if isinstance(C, pd.DataFrame):
        C1 = pd.DataFrame(C1, index=C.index, columns=C.columns)

    if return_eigenvector_ratio:
        return C1, u0.flatten().clip(0.0, 1.0), ev_ratio

    return C1, u0.flatten().clip(0.0, 1.0)  # TODO confirm clipping
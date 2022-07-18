import numpy as np
from .utils import (
    _filter_comparisons_by_ids,
    _group_comparisons_by_identity,
    _add_empty_mapping,
)


def compute_results_supervised(comparisons, groups=None):
    """
    Input:

        comparisons is the dict of provider comparison outputs over the subset
        of faces manually verified as a match to the seed identity.


    Output:

        We return a map of the form
        {
            'metrics' -> metrics with aggregate, person id, or group name as keys
            'extrema' -> {
                'same-id' -> same-id extrema with same keys as above
                'diff-id' -> diff-id extrema with same keys as above
            },
            'matrix' -> same-id comparison matrix with person id as keys,
            'face_to_person' -> map from face id to person id
        }


    Output explained:

        For the aggregate, each person, and any specified groups we compute

        (a) metrics of the form
        {
            true_match_confidences -> match confidences for same id pairs,
            no_match_confidences -> match confidences for diff id pairs,
            all_confidence -> all match confidences,
            ts -> thresholds used to compute FMR and FNMR,
            false_match_rate -> FMR,
            false_non_match_rate -> FNMR,
        }

        (b) same-id and diff-id extrema of the form
        {
            raw -> pairs of face comparisons sorted by confidence
            avg -> faces sorted by avg confidence in all comparisons
            median -> faces sorted by median confidence in all comparisons

            ** same-id is sorted descending by confidence *
            ** diff-id is sorted ascending by confidence **
        }

        We also compute
        (c) a matrix of confidences for same-id comparisons
            (to run MDS on each identity) that is F x F for F faces.
            It has a 1 on the diagonal and the comparison output
            for faces i,j in the i,jth entry.

        (d) a map from face-id to person-id (for convenience later on).
    """
    comparisons_per_person = _group_comparisons_by_identity(comparisons)

    metrics = {}
    metrics["aggregate"] = _compute_metrics_for_labeled_faces(comparisons)
    for person_id, sub_comparisons in comparisons_per_person.items():
        metrics[person_id] = _compute_metrics_for_labeled_faces(sub_comparisons)

    if groups is not None:
        for label, ids in groups.items():
            sub_comparisons = _filter_comparisons_by_ids(comparisons, ids)
            metrics[label] = _compute_metrics_for_labeled_faces(sub_comparisons)

    extrema = {}
    extrema["aggregate"] = {
        "same_id": _sort_same_id_confidences(comparisons),
        "diff_id": _sort_diff_id_confidences(comparisons),
    }
    for person_id, sub_comparisons in comparisons_per_person.items():
        extrema[person_id] = {
            "same_id": _sort_same_id_confidences(sub_comparisons),
            "diff_id": _sort_diff_id_confidences(sub_comparisons, for_person=person_id),
        }

    matrix = {}
    for person_id, sub_comparisons in comparisons_per_person.items():
        matrix[person_id] = _load_similarity_matrix(sub_comparisons, same_id=True)

    face_to_person = {}
    for (face1_id, person1_id), (face2_id, person2_id) in comparisons:
        face_to_person[face1_id] = person1_id
        face_to_person[face2_id] = person2_id

    return {
        "metrics": metrics,
        "extrema": extrema,
        "matrix": matrix,
        "face_to_person": face_to_person,
    }


def _load_similarity_matrix(comparisons, same_id=False):
    """
    Returns a matrix M with M[i][j] = confidence from comparison(face i, face j).
    """
    face_ids = set()
    for (face1_id, person1_id), (face2_id, person2_id) in comparisons:
        if same_id and person1_id != person2_id:
            continue
        face_ids.add(face1_id)
        face_ids.add(face2_id)
    face_ids = list(face_ids)

    face_ids_to_idx = {}
    for i, face_id in enumerate(face_ids):
        face_ids_to_idx[face_id] = i

    matrix = [[-1 for _ in range(len(face_ids))] for _ in range(len(face_ids))]
    for key in comparisons:
        (face1_id, person1_id), (face2_id, person2_id) = key
        if same_id and person1_id != person2_id:
            continue
        i1 = face_ids_to_idx[face1_id]
        i2 = face_ids_to_idx[face2_id]
        matrix[i1][i2] = comparisons[key]
        matrix[i2][i1] = comparisons[key]

    for i in range(len(face_ids)):
        matrix[i][i] = 1

    for ri, r in enumerate(matrix):
        for ci, c in enumerate(r):
            if c == -1:
                return {"data": None, "ids": None}

    return {
        "data": np.array(matrix),
        "ids": face_ids,
    }


def _sort_diff_id_confidences(comparisons, for_person=None):
    """
    Returns a map
    {
        'raw'     ->  a list [(confidence, ((face1_id, person1_id), (face2_id, person2_id)))]
                      sorted by confidence (descending),
        'avg'     ->  a list [(avg_confidence, person_id, face_id, num_comparisons)] sorted
                      by average confidence over pairs with that face (descending),
        'median'  ->  same as 'avg' but for the median confidence,
    }

    Arg `for_person` allows us to solely consider comparisons including some person.
    """
    face_to_person_id = {}
    face_to_confidences = {}
    pairs_confidence_list = []
    for key, confidence in comparisons.items():
        ((face1_id, person1_id), (face2_id, person2_id)) = key
        if person1_id == person2_id:
            continue

        if for_person is None or for_person == person1_id:
            _add_empty_mapping(face1_id, face_to_confidences, list=True)
            face_to_confidences[face1_id].append(confidence)
            face_to_person_id[face1_id] = person1_id

        if for_person is None or for_person == person2_id:
            _add_empty_mapping(face2_id, face_to_confidences, list=True)
            face_to_confidences[face2_id].append(confidence)
            face_to_person_id[face2_id] = person2_id

        pairs_confidence_list.append((confidence, key))

    pairs_confidence_list.sort(reverse=True)

    face_average_confidences = [
        (np.mean(confidences), face_to_person_id[face_id], face_id, len(confidences))
        for face_id, confidences in face_to_confidences.items()
    ]
    face_average_confidences.sort(reverse=True)

    face_median_confidences = [
        (np.median(confidences), face_to_person_id[face_id], face_id, len(confidences))
        for face_id, confidences in face_to_confidences.items()
    ]
    face_median_confidences.sort(reverse=True)

    return {
        "raw": pairs_confidence_list,
        "avg": face_average_confidences,
        "median": face_median_confidences,
    }


def _sort_same_id_confidences(comparisons):
    """
    Returns a map
    {
        'raw'     ->  a list [(confidence, person_id, face1_id, face2_id)] sorted by
                      confidence (ascending),
        'avg'     ->  a list [(avg_confidence, person_id, face_id, num_comparisons)] sorted
                      by average confidence over pairs with that face (ascending),
        'median'  ->  same as 'avg' but for the median confidence,
    }

    Arg `for_person` allows us to solely consider comparisons including some person.
    """
    face_to_confidences = {}
    face_to_person = {}
    pairs_confidence_list = []
    for (
        (face1_id, person1_id),
        (face2_id, person2_id),
    ), confidence in comparisons.items():
        if person1_id != person2_id:
            continue

        assert face1_id != face2_id
        _add_empty_mapping(face1_id, face_to_confidences, list=True)
        _add_empty_mapping(face2_id, face_to_confidences, list=True)

        face_to_person[face1_id] = person1_id
        face_to_person[face2_id] = person2_id

        face_to_confidences[face1_id].append(confidence)
        face_to_confidences[face2_id].append(confidence)

        pairs_confidence_list.append((confidence, person1_id, face1_id, face2_id))

    pairs_confidence_list.sort()

    face_average_confidences = [
        (np.mean(confidences), face_to_person[face_id], face_id, len(confidences))
        for face_id, confidences in face_to_confidences.items()
    ]
    face_average_confidences.sort()

    face_median_confidences = [
        (np.median(confidences), face_to_person[face_id], face_id, len(confidences))
        for face_id, confidences in face_to_confidences.items()
    ]
    face_median_confidences.sort()

    return {
        "raw": pairs_confidence_list,
        "avg": face_average_confidences,
        "median": face_median_confidences,
    }


def _compute_metrics_for_labeled_faces(comparisons):
    """
    Computes FMR and FNMR given comparison output for annotated faces.
    """
    confidences_output = _compute_categorized_confidences(comparisons)
    true_match_confidences, no_match_confidences, all_confidences = confidences_output

    rates_output = _compute_match_rates_optimized(
        true_match_confidences, no_match_confidences
    )
    ts, false_match_rate, false_non_match_rate = rates_output

    return {
        "true_match_confidences": true_match_confidences,
        "no_match_confidences": no_match_confidences,
        "all_confidences": all_confidences,
        "ts": ts,
        "false_match_rate": false_match_rate,
        "false_non_match_rate": false_non_match_rate,
    }


def _compute_categorized_confidences(comparisons):
    """
    Groups the confidence output from a cloud provider
    by same-id pairs and diff-id pairs.
    """
    no_match_confidences = []
    true_match_confidences = []
    all_confidences = []
    for ((_, person1_id), (_, person2_id)), confidence in comparisons.items():
        if person1_id == person2_id:
            true_match_confidences.append(confidence)
        else:
            no_match_confidences.append(confidence)
        all_confidences.append(confidence)

    true_match_confidences = np.array(true_match_confidences)
    no_match_confidences = np.array(no_match_confidences)
    all_confidences = np.array(all_confidences)
    return true_match_confidences, no_match_confidences, all_confidences


def _compute_match_rates(true_match_confidences, no_match_confidences, delta=0.001):
    """
    Computes FMR and FNMR.

    Uses a slow O(TN) algorithm, for T timesteps and N confidences.
    """
    ts = []
    false_non_match_count = []
    false_match_count = []
    t = 0
    # Include a small fraction of the delta in the upper bound
    # to account for float precision.
    while t <= 1 + delta / 2:
        ts.append(t)
        false_match_count.append(sum(no_match_confidences >= t))
        false_non_match_count.append(sum(true_match_confidences < t))
        t += delta

    ts = np.array(ts)
    false_match_rate = np.array(false_match_count) / len(no_match_confidences)
    false_non_match_rate = np.array(false_non_match_count) / len(true_match_confidences)
    return ts, false_match_rate, false_non_match_rate


def _compute_match_rates_optimized(
    true_match_confidences, no_match_confidences, delta=0.001
):
    """
    Computes FMR and FNMR.

    Uses a faster O(T + N log N) algorithm, for T timesteps and N confidences.

    This should be an optimization if T >> log N.
    """
    ts = []
    false_non_match_count = []
    false_match_count = []

    true_match_confidences = np.sort(true_match_confidences)
    no_match_confidences = np.sort(no_match_confidences)

    num_no_matches = len(no_match_confidences)
    t = 0
    # true_i is the greatest index i such that true_match_confidences[i] >= curr_t
    # false_i is the greatest index i such that no_match_confidences[i] >= curr_t
    true_i = 0
    false_i = 0
    # Include a small fraction of the delta in the upper bound
    # to account for float precision.
    while t <= 1 + delta / 2:
        ts.append(t)
        while (
            true_i < len(true_match_confidences) and true_match_confidences[true_i] < t
        ):
            true_i += 1
        while false_i < len(no_match_confidences) and no_match_confidences[false_i] < t:
            false_i += 1
        false_match_count.append(num_no_matches - false_i)
        false_non_match_count.append(true_i)
        t += delta

    ts = np.array(ts)

    if len(no_match_confidences) > 0:
        false_match_rate = np.array(false_match_count) / len(no_match_confidences)
    else:
        false_match_rate = None
        raise Exception("Insufficient data to compute false match rate")
    if len(true_match_confidences) > 0:
        false_non_match_rate = np.array(false_non_match_count) / len(
            true_match_confidences
        )
    else:
        false_non_match_rate = None
        raise Exception("Insufficient data to compute false non-match rate")

    return ts, false_match_rate, false_non_match_rate

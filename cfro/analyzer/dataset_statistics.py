def compute_dataset_statistics(unsupervised_comparisons, supervised_comparisons):
    """
    Computes statistics like
    * number of identities
    * faces collected per id
    * correct faces collected per id
    * number of same/diff id comparisons
    """
    total_faces_per_id = _count_faces_per_id(unsupervised_comparisons)
    num_same_seed_id_pairs, num_diff_seed_id_pairs = _count_same_diff_pairs(
        unsupervised_comparisons
    )

    correct_faces_per_id = _count_faces_per_id(supervised_comparisons)

    correct_faces_percentages_per_id = {}
    for person_id, total_faces in total_faces_per_id.items():
        if person_id in correct_faces_per_id:
            correct_faces_percentages_per_id[person_id] = (
                100 * correct_faces_per_id[person_id] / total_faces
            )
        else:
            correct_faces_percentages_per_id[person_id] = None

    num_labeled_same_id_pairs, num_labeled_diff_id_pairs = _count_same_diff_pairs(
        supervised_comparisons
    )

    return {
        "num_identities": len(total_faces_per_id),
        "total_faces_per_id": total_faces_per_id,
        "correct_faces_per_id": correct_faces_per_id,
        "correct_faces_percentages_per_id": correct_faces_percentages_per_id,
        "num_comparisons": {
            "unlabeled-same-seed": num_same_seed_id_pairs,
            "unlabeled-diff-seed": num_diff_seed_id_pairs,
            "labeled-same-id": num_labeled_same_id_pairs,
            "labeled-diff-id": num_labeled_diff_id_pairs,
        },
    }


def _count_faces_per_id(comparisons):
    """
    Counts the number of distinct faces per seed id,
    given a map of the pairs of face comparisons.
    """
    person_id_to_face_set = {}
    for key in comparisons:
        for (face_id, person_id) in key:
            if person_id not in person_id_to_face_set:
                person_id_to_face_set[person_id] = set()
            person_id_to_face_set[person_id].add(face_id)

    person_id_to_face_count = {}
    for person_id, faces in person_id_to_face_set.items():
        person_id_to_face_count[person_id] = len(faces)
    return person_id_to_face_count


def _count_same_diff_pairs(comparisons):
    """
    Counts the number of face comparisons from the
    same seed id and from different seed ids.
    """
    same_id_pairs = 0
    for ((_, person1_id), (_, person2_id)) in comparisons:
        same_id_pairs += int(person1_id == person2_id)
    diff_id_pairs = len(comparisons) - same_id_pairs
    return same_id_pairs, diff_id_pairs

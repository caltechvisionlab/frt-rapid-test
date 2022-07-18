def _filter_comparisons_by_ids(comparisons, ids):
    """
    Returns comparisons where both seed people
    belong to the iterable of ids.
    """
    safe_comparisons = {}
    for key, value in comparisons.items():
        (_, person1_id), (_, person2_id) = key
        if person1_id in ids and person2_id in ids:
            safe_comparisons[key] = value

    return safe_comparisons


def _group_comparisons_by_identity(comparisons):
    """
    Returns a map from each person to the subset
    of comparisons involving that person.
    """
    person_to_comparisons = {}

    for key, confidence in comparisons.items():
        ((_, person1_id), (_, person2_id)) = key
        _add_empty_mapping(person1_id, person_to_comparisons, dict=True)
        person_to_comparisons[person1_id][key] = confidence
        if person1_id != person2_id:
            _add_empty_mapping(person2_id, person_to_comparisons, dict=True)
            person_to_comparisons[person2_id][key] = confidence

    return person_to_comparisons


def _add_empty_mapping(key, map, list=False, dict=False):
    if key not in map:
        if list:
            map[key] = []
        if dict:
            map[key] = {}

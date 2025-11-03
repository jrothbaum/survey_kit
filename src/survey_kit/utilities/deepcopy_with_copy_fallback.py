import copy


def deepcopy_with_fallback(obj):
    # Try standard deepcopy first
    try:
        result = copy.deepcopy(obj)
        return result
    except (TypeError, AttributeError):
        result = copy.copy(obj)

    return result

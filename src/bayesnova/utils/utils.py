import numpy as np

def map_array_to_dict(
    array: np.ndarray,
    keys: list[str]
):
    """Map an array to a dictionary.

    Args:
        array (np.ndarray): Array to map.
        keys (list[str]): Keys to map to.

    Returns:
        dict: Dictionary of mapped values.
    """

    return dict(zip(keys, array))

def map_dict_to_array(
    dict: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Map a dictionary to an array.

    Args:
        dict (dict[str, float]): Dictionary to map.

    Returns:
        tuple[np.ndarray, list[str]]: Tuple of mapped array and keys.
    """

    keys = list(dict.keys())
    values = np.array([dict[key] for key in keys])

    return values, keys
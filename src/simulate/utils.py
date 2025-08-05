"""Utility functions for quantum error correction simulations.

This module provides helper functions for data manipulation, array processing,
and common operations used throughout the simulation package.
"""

from typing import Any, Iterable, List, Tuple, Union
import warnings

import numpy as np

def flatten(lst: Union[Any, Iterable, np.ndarray]) -> List[Any]:
    """Flatten nested lists and arrays into a single list.
    
    Args:
        lst: Input data to flatten (can be nested lists, tuples, arrays, or scalars)
        
    Returns:
        Flattened list containing all elements
    """
    if isinstance(lst, np.ndarray):
        return lst.flatten().tolist()
    # Handle scalar inputs by wrapping them in a list
    if not isinstance(lst, Iterable):
        return [lst]
    ret = []
    for x in lst:
        if isinstance(x, (list, tuple, np.ndarray)):
            ret.extend(flatten(x))
        else:
            ret.append(x)
    return ret
def groupby(arr: np.ndarray, by: Union[str, List[str]]) -> Tuple[
        np.ndarray, List[np.ndarray]]:
    """Group a structured array by specified columns.
    
    Args:
        arr: Structured numpy array to group
        by: Column name(s) to group by
        
    Returns:
        Tuple of (unique values, list of grouped arrays)
    """
    arr = np.sort(arr, order=by)
    unq, idx = np.unique(arr[by], return_index=True)
    return unq, np.split(arr, idx[1:])

def reorder(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the permutation indices to reorder y to match x.
    
    Args:
        x: Target array order
        y: Array to be reordered
        
    Returns:
        Permutation indices such that y[indices] equals x
        
    Raises:
        ValueError: If y is not a permutation of x
    """
    if not np.array_equal(np.sort(x), np.sort(y)):
        raise ValueError("y is not a permutation of x.")
    idx_x_sorted = np.argsort(x)
    idx_y_sorted = np.argsort(y)
    rank_of_original_x_index = np.empty_like(idx_x_sorted)
    rank_of_original_x_index[idx_x_sorted] = np.arange(len(x))
    return idx_y_sorted[rank_of_original_x_index]

def get_index(d: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Get indices of elements x in array d.
    
    Args:
        d: Array to search in
        x: Elements to find indices for
        
    Returns:
        Array of indices such that d[indices] equals x
        
    Note:
        Assumes all entries of x are present in d.
    """
    if not np.all(np.isin(x, d)):
        warnings.warn(
            f"get_index: x contains values not in d: {x[~np.isin(x, d)]}")
    orig_shape = x.shape
    order = np.argsort(d)  # permutation that would sort d
    d_sorted = d[order]  # the sorted view of d
    pos_sorted = np.searchsorted(d_sorted, x.flatten())  # positions in sorted view
    return order[pos_sorted].reshape(orig_shape)  # indices in the original d

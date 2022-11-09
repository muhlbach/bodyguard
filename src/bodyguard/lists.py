#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import itertools
import collections
import numpy as np        
import pandas as pd
import numbers
from typing import List


# User
from .sanity_check import check_type
from .tools import remove_empty_elements
from .exceptions import print_warning

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def listify(x):
    """ Robust way to convert x to list"""    
    if isinstance(x, list):
        pass
    elif isinstance(x, (np.ndarray, pd.Series)):
        x = x.tolist()
        if pd.api.types.is_number(x):
            x = [x]
    elif isinstance(x,(str,numbers.Number)):
        x = [x]
    elif isinstance(x,tuple):
        x = [i for i in x]
    elif x is None:
        x = [x]
    else:
        print_warning(f"Unknown dtype: {type(x)} for argument 'x': {x}")
        try:
            x = [i for i in x]
        except:
            x = [x]
            
    return x

def intersect(l, l_other=None):
    """
    Find intersection of elements in list (typically list of lists)
    """
    check_type(x=l,allowed=list,name="l")

    if l_other is not None:
        check_type(x=l_other,allowed=list,name="l_other")
        
        l_intersect = list(set(l) & set(l_other))
    else:    
        l_intersect = list(set.intersection(*map(set,l)))
    
    return l_intersect
    
def subset(l, l_boolean):
    """
    Subset list based on another boolean list
    """
    check_type(x=l,allowed=list,name="l")
    check_type(x=l_boolean,allowed=list,name="l_boolean")
    
    if not len(l)==len(l_boolean):
        raise Exception(f"Length of 'l' (={len(l)}) must equal length of 'l_boolean' (={len(l_boolean)})")
    
    l_subset = list(itertools.compress(l, l_boolean))
    
    return l_subset

def unique(l):
    """
    Subset to unique elements of list while preserving order
    
    NB: Elements cannot be unhasable types (e.g., dict)
    """
    check_type(x=l,allowed=list,name="l")
    
    # Exclude duplicates but preserve order
    l_unique = list(dict.fromkeys(l))
    
    return l_unique

def drop_duplicates(l):
    """
    Verbatim unique()
    """
    return unique(l=l)
    
def flatten_iterator(l):
    """
    Return iterator over flat list
    """
    for i in l:
            if isinstance(i, collections.Iterable) and not isinstance(i, str):
                for subc in flatten_iterator(i):
                    yield subc
            else:
                yield i

def flatten(l):
    """
    Flatten list
    """
    check_type(x=l,allowed=list,name="l")
    
    l_flat = list(flatten_iterator(l))
    
    return l_flat


def flatten_list(_list: List[list]) -> list:
    """Merge/flatten a list of lists into one single list.

    Args:
        _list (List[list]): List of lists to be merged/flattened

    Returns:
        list: Flattened list
    """
    return [item for sublist in _list for item in sublist]


def remove_empty(l):
    """
    Remove empty elements
    """
    check_type(x=l,allowed=list,name="l")
    
    l_clean = remove_empty_elements(x=l)
    
    return l_clean

def remove(l, l_remove):
    check_type(x=l,allowed=list,name="l")
    check_type(x=l_remove,allowed=list,name="l_remove")
    
    l_keep = [x for x in l if x not in l_remove]
    
    return l_keep    
    
def not_in(l_parent, l_child):
    """
    Return elements in "l_parent" not in "l_child"
    """
    check_type(x=l_parent,allowed=list,name="l_parent")
    check_type(x=l_child,allowed=list,name="l_child")

    l_notin = [x for x in l_parent if x not in l_child]
    
    return l_notin

def uncommon(l1, l2):
    """
    Return elements that are not common (opposite of intersection)
    """
    check_type(x=l1,allowed=list,name="l1")
    check_type(x=l2,allowed=list,name="l2")

    l1_uncommon = not_in(l_parent=l1, l_child=l2)
    l2_uncommon = not_in(l_parent=l2, l_child=l1)
    
    l_uncommon = l1_uncommon+l2_uncommon

    return l_uncommon

    
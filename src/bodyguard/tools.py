#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
import numbers
from functools import reduce

# User
from .exceptions import WrongInputException
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def convert_to_list(x):
    """Convert object "x" to list"""
    
    # List non-iterables and iterables. Note that we cannot rely on __iter__ attribute, because then str will be iterable!
    NON_ITERABLES = (str,numbers.Number)
    ITERABLES = (list, tuple, np.ndarray, pd.Series)
    
    if isinstance(x,NON_ITERABLES):
        x = [x]
    elif isinstance(x,ITERABLES):
        x = [i for i in x]
    else:
        try:
            x = [i for i in x]
        except:
            x = [x]
            
    return x

def isin(a, b, how="all", return_element_wise=True):
    """Check if any/all of the elements in 'a' is included in 'b'
    Note: Argument 'how' has NO EFFECT when 'return_element_wise=True'
    
    """
    ALLOWED_HOW = ["all", "any"]
    
    if how not in ALLOWED_HOW:
        raise WrongInputException(input_name="how",
                                  provided_input=how,
                                  allowed_inputs=ALLOWED_HOW)

    # Convert "a" and "b" to lists
    a = convert_to_list(x=a)
    b = convert_to_list(x=b)

    # For each element (x) in a, check if it equals any element (y) in b
    is_in_temp = [any(x == y for y in b) for x in a]

    if return_element_wise:
        is_in = is_in_temp
    else:
        # Evaluate if "all" or "any" in found, when we only return one (!) answer
        if how=="all":
            is_in = all(is_in_temp)
        elif how=="any":
            is_in = any(is_in_temp)
                    
    if (len(a)==1) and isinstance(is_in, list):
        # Grab first and only argument if "a" is not iterable
        is_in = is_in[0]
            
    return is_in

def remove_empty_elements(d):
    """
    Remove empty elements from dict or list
    """
    if isinstance(d, dict):
        return dict((k, remove_empty_elements(v)) for k, v in d.items() if v and remove_empty_elements(v))
    elif isinstance(d, list):
        return [remove_empty_elements(v) for v in d if v and remove_empty_elements(v)]
    else:
        return d
    
    
def merge_multiply_dfs(l,
                       how='inner',
                       on=None,
                       left_on=None,
                       right_on=None,
                       left_index=False,
                       right_index=False,
                       sort=False,
                       suffixes=('_x', '_y'),
                       copy=True,
                       indicator=False,
                       validate=None):
    """
    Merge multiple pd.DataFrame
    """
    # Check if l is a list
    if not isinstance(l, list):
        raise WrongInputException(input_name="l",
                                  provided_input=l,
                                  allowed_inputs=["list"])
        
    # Check if all elements are pd.dataframes
    if not all(isinstance(df,pd.DataFrame) for df in l):
        raise Exception("All elements in 'l' must be instances of pd.DataFrame")
        
    # Merge
    df_merged = reduce(lambda left,right: pd.merge(left,
                                                   right,
                                                   how=how,
                                                   on=on,
                                                   left_on=left_on,
                                                   right_on=right_on,
                                                   left_index=left_index,
                                                   right_index=right_index,
                                                   sort=sort,
                                                   suffixes=suffixes,
                                                   copy=copy,
                                                   indicator=indicator,
                                                   validate=validate
                                                   ), l)
    
    return df_merged


        








    
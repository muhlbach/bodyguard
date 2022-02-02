#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np

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
    
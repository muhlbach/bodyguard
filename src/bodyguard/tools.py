#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import sys, time, random
from datetime import datetime
import numpy as np
import pandas as pd
import numbers
from functools import reduce

# User
from .exceptions import WrongInputException
from .sanity_check import check_type
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def initialize_script(seed=1991):

    # Start Timer
    t0 = time.time()

    # Set seed
    random.seed(1991)

    # Get current time and date
    now = datetime.now()
    
    print2(f"This script was initialized {now.strftime('at %H:%M:%S on %B %d, %Y')}")

    return t0


def end_script(t0):

    # Stop Timer
    t1 = time.time()
    
    # Set seed
    random.seed(1991)

    # Get current time and date
    now = datetime.now()
    
    print2(f"""This script was ended {now.strftime('at %H:%M:%S on %B %d, %Y')}:

   It took:
       {int(t1-t0)} seconds,
       {int((t1-t0)/60)} minutes, and
       {round((t1-t0)/3600,2)} hours
    """)
    
def stop():
    sys.exit("Script was stopped by user")    


def downcast(df, downcast_int=True, downcast_float=True):
    """
    Downcast numerical dtypes of dataframe
    """
    check_type(x=df,
               allowed=pd.DataFrame,
               name="df")
        
    if downcast_int:
        # Downcast to integer if possible
        df = df.apply(lambda x: pd.to_numeric(x, downcast='integer', errors="ignore"))

    if downcast_float: 
        # Downcast float if possible
        df = df.apply(lambda x: pd.to_numeric(x, downcast='float', errors="ignore"))
    
    return df


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
    HOW_OPT = ["all", "any"]

    # NB! We cannot use ".sanity_check.check_str" here, because it will lead to RecursionError: maximum recursion depth exceeded while calling a Python object
    # This is because ".sanity_check.check_str" itself uses isin.
    if not any(how==h for h in HOW_OPT):
        raise WrongInputException(x=how,
                                  allowed=HOW_OPT,
                                  name="how")
    
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
    check_type(x=l, allowed=list,name="l")
    
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




def print2(msg=""):
    
    print(
f"""
------------------------- MESSAGE -------------------------
{msg}
-----------------------------------------------------------
"""
        )
        
def insert_missing_rows_groupwise(df, groups, df_insert=None):
    """
    Insert rows with missing groups.
    E.g., if year 1 has observations 'a', and 'b', whereas year 2 has 'b' and 'c', then the merged data with have 'a', 'b', and 'c' for both year 1 and 2.
    """    
    # Pre-allocate list of mergeables
    mergeables = []
    
    for g in groups:
        # Get df of unique values
        df_mergeable = df[g].drop_duplicates().reset_index(drop=True)
        
        # Add key
        df_mergeable['key'] = "A"
        
        # Add to list of mergeable dfs
        mergeables.append(df_mergeable)
        
    if df_insert is not None:        
        # Add key to df to be inserted
        df_insert['key'] = "A"
            
        # Add to list of mergeable dfs
        mergeables.append(df_insert.drop_duplicates().reset_index(drop=True))
    
    # Get the number of unique rows needed
    nunique_rows = np.prod([len(d) for d in mergeables])

    # Construct data with all groups
    df_unique = merge_multiply_dfs(l=mergeables,on=['key'],how='outer') 

    # Remove key
    df_unique.drop(columns="key", inplace=True)
    
    # Sanity check
    if len(df_unique)!=nunique_rows:
        raise Exception("Creating the unique data to be merged with didn't work.")

    # Merge with unique data and thereby insert extra rows
    df_merged = df.merge(right=df_unique,
                         on=df_unique.columns.tolist(),
                         how="outer")
    
    return df_merged







    
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import numpy as np
from .sanity_check import check_type
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def convert_dict_to_df(x):
    """ Converts a dictionary to a pandas.DataFrame """
    check_type(x=x,allowed=dict)    
    
    try:
        x = pd.DataFrame().from_dict(data=x,orient="index")
    except ValueError:
        # This happens because we have already named the index
        keys, dfs = list(x.keys()), list(x.values())
        
        # Concat all dataframes
        x = pd.concat(objs=dfs, axis=0, ignore_index=True, sort=False)
        
        # Overwrite index
        x.index = keys
        
    return x
    
def convert_df_to_dict(x):
    """ Converts a pandas.DataFrame to a dict"""
    check_type(x=x,allowed=pd.DataFrame)        

    x = x.to_dict(orient="index")
    x = {k: pd.DataFrame().from_dict(data=v, orient="index").T.rename(index={0:k}) for k,v in x.items()}
    
    return x

def convert_to_df(x):
    """ Converts whatever to pandas.DataFrame"""
    check_type(x=x,
               allowed=(pd.DataFrame,pd.Series,np.ndarray)
               )

    # Break dependency
    x = x.copy()
    
    if not isinstance(x, pd.DataFrame):
        if isinstance(x, pd.Series):
            if x.name is None:
                x.name = "X1"
             
            # series to frame
            x = x.to_frame()
        elif isinstance(x, np.ndarray):
            try:
                x.shape[1]
            except IndexError:
                x = x.reshape(-1,1)
                
            # convert ndarray to frame
            x = pd.DataFrame(data=x,
                             coluns=[f"X{j}" for j in range(1,x.shape[1]+1)])
                
        else:
            raise Exception(f"X must be either pd.Series or np.ndarray but is {type(x)}")
        
    return x
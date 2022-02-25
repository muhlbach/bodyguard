#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
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

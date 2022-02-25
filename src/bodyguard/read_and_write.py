#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import os
import numpy as np
import pandas as pd
import pickle
from .sanity_check import check_str, check_type
from .strings import contains_number
from .tools import isin, downcast
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def read_file(path, clean_up=True, downcast_float=True, **kwargs):
    """
    Read data by pandas
    """
    check_type(x=path,allowed=str,name="path")

    # Split path to identify extension of gile
    path_split = path.rsplit(".", maxsplit=1)
    
    # Obtain file extension to guide file reading
    f_ext = path_split[-1]
    
    check_str(x=f_ext,allowed=["csv","xls", "xlsx"])
    
    if f_ext=="csv":
        df = pd.read_csv(filepath_or_buffer=path, **kwargs)
    elif isin(a=f_ext,b=["xls", "xlsx"]):
        df = pd.read_excel(io=path, **kwargs)
        
    if clean_up:
        
        # Subset to complete data
        mask_na = df.isna().any(axis=1)
        df = df.loc[~mask_na]

        # Find integers in order to convert to float (which is the richest numerical dtype)
        mask_int = df.apply(lambda x: x.map(type)==int)

        # Convert int to float
        for c in mask_int.columns:
            df.loc[mask_int[c],c] = df.loc[mask_int[c],c].astype(float)
            
        # List indices
        idx = df.index.tolist()
        
        # Initialize
        cnt = 0
        is_inconsistent = True
        while is_inconsistent:
            
            # Check if multiple dtypes exist
            if df.loc[idx[cnt]:].apply(lambda x: x.map(type).nunique()==1).all():
                is_inconsistent = False
            else:
                cnt += 1
    
        # Find col names
        cols = df.loc[idx[cnt-1]].tolist()
        
        # Subset new data
        df = pd.DataFrame(data=df.loc[idx[cnt]:].values,
                          columns=cols)
        
        # Downcast where posssible
        # We fix "downcast_int" because we have upcasted integers to floats. But we allow user to decide whether or not to downcast floats.
        df = downcast(df=df, downcast_int=True, downcast_float=downcast_float)
        
    return df


def pd_to_parquet(df,path,n_files=10,engine='auto',compression='BROTLI',index=None,partition_cols=None):
    """
    Save (multiple) pandas DataFrames as parquet files
    """
    # Sanity checks
    check_type(x=df, allowed=pd.DataFrame)
    check_type(x=n_files, allowed=int)
    
    if path.count(".")!=1:
        raise Exception(f"Argument 'path' is allowed to contain 1 dot (.) but it contains {path.count('.')}")

    # Split path to keep file extension
    path_split = path.split(".")
    
    # Split df into multiple dfs
    dfs = np.array_split(ary=df,
                            indices_or_sections=n_files,
                            axis=0)
    
    for i,d in enumerate(dfs):

        # Filename
        path_i = path_split[0]+f"_{i}."+path_split[1]

        # Save multiple
        d.to_parquet(path=path_i,
                        engine=engine,
                        compression=compression,
                        index=index,
                        partition_cols=partition_cols)


def pd_from_parquet(path,engine='auto',columns=None,storage_options=None,use_nullable_dtypes=False, verbose=False):
    """
    Read (multiple) parquet files as pandas DataFrames
    """        
    check_type(x=path,allowed=str,name="path")
    
    # Split path to keep file extension
    path_split = path.rsplit(".", maxsplit=1)

    # File extension
    file_ext = path_split[-1:][0]
    
    # Get directory part of path and split
    dir_split = path_split[0].rsplit("/", maxsplit=1)
    
    # Join bits to form path of directory
    dir_path = dir_split[0]

    # Parent file name
    file_name = dir_split[1]
            
    # List all files in directory of parent file
    files_all = os.listdir(dir_path)
    
    if verbose>1:
        print(f"Files found in path: \n{files_all}")

    # List only relevant files; They must contain both file name and file extension as well as at least one number
    files = [f for f in files_all if all(c in f for c in [file_name,file_ext]) and contains_number(s=f)]
    
    # Sort
    files.sort()
    
    if verbose:
        print(f"Relevant files found in path: \n{files}")

    # Pre-allocate
    dfs = []
    
    for f in files:

        # File-specific path
        path_f = os.path.join(dir_path,f)

        # Read file
        df_temp = pd.read_parquet(path=path_f,
                                    engine=engine,
                                    columns=columns,
                                    storage_options=storage_options,
                                    use_nullable_dtypes=use_nullable_dtypes)
        # Append
        dfs.append(df_temp)
        
    # Concatenate dfs from list of dfs
    df = pd.concat(objs=dfs,
                    axis=0,
                    join='outer',
                    ignore_index=False,
                    keys=None,
                    levels=None,
                    names=None,
                    verify_integrity=False,
                    sort=False,
                    copy=True)

    return df

# Save and load objects using pickle
def save_object_by_pickle(filename,obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj=obj, file=f, protocol=-1)

def load_object_by_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(file=f)

        
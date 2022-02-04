#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def pd_to_parquet(df,path,n_files=10,engine='auto',compression='BROTLI',index=None,partition_cols=None):
    """
    Save (multiple) pandas DataFrames as parquet files
    """
    # Sanity checks
    if not isinstance(df, pd.DataFrame):
        raise Exception(f"Argument 'df' is not instance of pd.DataFrame. It is type: {type(df)}")
        
    if not isinstance(n_files,int):
        raise Exception(f"Argument 'n_files' is not instance of int. It is type: {type(n_files)}")

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


def pd_from_parquet(path,engine='auto',columns=None,storage_options=None,use_nullable_dtypes=False):
    """
    Read (multiple) parquet files as pandas DataFrames
    """        
    # Split path to keep file extension
    path_split = path.split(".")
    
    # File extension
    file_ext = path_split[-1:]
    if len(file_ext)==1:
        file_ext = file_ext[0]
    else:
        raise Exception(f"Internally created 'file_ext' is not of len 1, but is {file_ext}")
    
    # Get directory part of path and split
    dir_split = path_split[0].split("/")
    
    # Join bits to form path of directory
    dir_path = "/".join(dir_split[:-1])

    # Parent file name
    file_name = dir_split[-1:]
    if len(file_name)==1:
        file_name = file_name[0]
    else:
        raise Exception(f"Internally created 'filename' is not of len 1, but is {file_name}")
            
    # List all files in directory of parent file
    files_all = os.listdir(dir_path)
    
    # List only relevant files; They must contain both file name and file extension as well as at least one number
    files = [f for f in files_all if all(c in f for c in [file_name,file_ext]) and bd.strings.contains_number(s=f)]
    
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

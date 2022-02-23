#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import numpy as np
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def Lk_norm_distance(X,Y,k=1/10,axis=1):
    """
    Compute the L_k norm distance between two dataframes
    Default k=1/10 inspired by Aggarwal et al. (2001) "The Surprising Behaviour of Distance Metrics in High Dimensions"
    """
    # Breaks links
    X = X.copy()
    Y = Y.copy()

    # Convert types            
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = X.to_frame()
            if axis==1:
                X = X.T
        else:
            X = pd.DataFrame(X)
        
    if not isinstance(Y, pd.DataFrame):
        if isinstance(Y, pd.Series):
            Y = Y.to_frame()
            if axis==1:
                Y = Y.T
        else:
            Y = pd.DataFrame(Y)        
        
    # Check dimensions
    if axis==0:
        if X.shape[0]!=Y.shape[0]:
            raise Exception(f"Rows of X and Y must match, but X has {X.shape[0]} rows and Y has {Y.shape[0]} rows")
    elif axis==1:
        if X.shape[1]!=Y.shape[1]:
            raise Exception(f"Columns of X and Y must match, but X has {X.shape[1]} columns and Y has {Y.shape[1]} columns")
        
    # Transpose
    if axis==0:
        X = X.T
        Y = Y.T

    # Pre-allocate    
    distances = []
    
    for row in Y.index:

        # Compute distance        
        distance_temp = X.subtract(Y.loc[row], axis=1).abs().pow(k).sum(axis=1).pow(1/k)
                         
        # Name the resulting series
        distance_temp.name = row
            
        # Append 
        distances.append(distance_temp)

    # Collect all distances
    df_distance = pd.concat(objs=distances,axis=1)
        
    return df_distance
    
    
def normalize_by_norm(x, norm="L2", axis=0):
    """
    Normalize x by the norm

    For instance, L2 norm means that sum(x^2)=1
    """
    # Break link
    x = x.copy()

    reverse_axis = np.where(axis==0,1,0).item()

    # Fix order
    if norm=="L1":
        norm_order=1
    elif norm=="L2":
        norm_order=2
    else:
        norm_order=norm
        
    # Normalize using the order
    if isinstance(x, list):
        x = [normalize_by_norm(x=elem,norm=norm) for elem in x]
    if isinstance(x, pd.DataFrame):
        x = x.div(np.linalg.norm(x=x, axis=axis, keepdims=False), axis=reverse_axis)
    else:
        x = x/np.linalg.norm(x=x,ord=norm_order, axis=None, keepdims=False)
    return x        
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from .sanity_check import check_type, check_str
from .tools import isin
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def compute_matrix_norm(x,which_norm="fro"):
    """
    Compute various matrix norms
    """
    ALLOWED_NORMS = ["fro", "nuc",
                     "spectral",
                     "sum_squares",
                     "inf",
                     "mse","rmse",
                     "all"]
    
    check_str(x=which_norm,allowed=ALLOWED_NORMS,name="which_norm")
        
    if which_norm in ["fro", "nuc"]:
        norm = np.linalg.norm(x=x, ord=which_norm)
    elif which_norm=="spectral":
        norm = np.linalg.norm(x=x, ord=2)
    elif which_norm=="sum_squares":
        norm = (x**2).sum()
    elif which_norm=="inf":
        norm = np.linalg.norm(x=x, ord=np.inf)
    elif which_norm=="rmse":
        norm = np.sqrt((x**2).mean())
    elif which_norm=="mse":
        norm = (x**2).mean()
        
    elif which_norm=="all":        
        norm = {
                "fro":np.linalg.norm(x=x, ord="fro"),
                "nuc":np.linalg.norm(x=x, ord="nuc"),
                "spectral":np.linalg.norm(x=x, ord=2),
                "sum_squares":(x**2).sum(),
                "inf":np.linalg.norm(x=x, ord=np.inf),
                "mse":(x**2).mean(),
                "rmse":np.sqrt((x**2).mean()),
                }
        
    return norm

def enforce_L2_normalization(x, axis=1):
    """Ensure that array has L2 norm normalized to 1"""
    if not all((x**2).sum(axis=axis).round(3)==1):
        x = normalize_by_norm(x=x,norm="L2",axis=axis)
    return x


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
    ALLOWED_TYPES = (list, pd.DataFrame,pd.Series,np.ndarray)
    check_type(x=x, allowed=ALLOWED_TYPES)

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
    elif isinstance(x, pd.DataFrame):
        if x.shape[0]==1:
            # This is a row-vector. Axis-argument is meaningless
            x = x.div(np.linalg.norm(x=x, axis=1, keepdims=False), axis=0)
        elif x.shape[1]==1:
            # This is a column-vector. Axis-argument is meaningless
            x = x.div(np.linalg.norm(x=x, axis=0, keepdims=False), axis=1)
        else:
            # This is a proper matrix
            x = x.div(np.linalg.norm(x=x, axis=axis, keepdims=False), axis=reverse_axis)
    elif isinstance(x, pd.Series):
        x = x/np.linalg.norm(x=x,ord=norm_order, axis=None, keepdims=False)
    elif isinstance(x,np.ndarray):
        x /= np.linalg.norm(x=x,ord=norm_order, axis=axis, keepdims=True)
        
    return x        


def compute_distance(a,b,metric="Lknorm", **kwargs):
    """
    Compute distance between two Dataframes or Series
    """
    #TODO: Implement np.ndarray
    
    # Settings
    TYPES_ALLOWED = (pd.Series,pd.DataFrame)
    SCIKIT_DISTANCE_METRIC = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                              'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
                              'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
                              'rogerstanimoto', 'russellrao', 'seuclidean',
                              'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    OTHER_DISTANCE_METRIC = ["Lknorm"]
    
    # Sanity checks
    check_type(x=a,allowed=TYPES_ALLOWED,name="a")
    check_type(x=b,allowed=TYPES_ALLOWED,name="b")
    check_str(x=metric,
              allowed=OTHER_DISTANCE_METRIC+SCIKIT_DISTANCE_METRIC,
              name="metric")
        
    # Convert types
    if isinstance(a, pd.Series):
        a = a.to_frame().T
    if isinstance(b, pd.Series):
        b = b.to_frame().T        

    # Compute distances
    if metric in SCIKIT_DISTANCE_METRIC:
        distances = pd.DataFrame(data=pairwise_distances(X=a,
                                                         Y=b,
                                                         metric=metric),
                                 index=a.index,
                                 columns=b.index)
                        
    elif metric=="Lknorm":
        distances = Lk_norm_distance(X=a,
                                     Y=b,
                                     **kwargs)
                                     
    return distances

def compute_similarity(a,b,metric="Lknorm",convert_to_angular_similarity=True,**kwargs):
    """
    Compute similarity between two Dataframes or Series
    """
    #TODO: Implement np.ndarray
    
    # Settings
    TYPES_ALLOWED = (pd.Series,pd.DataFrame)
    SCIKIT_DISTANCE_METRIC = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                              'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
                              'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
                              'rogerstanimoto', 'russellrao', 'seuclidean',
                              'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    CORRWITH_SIMILARITY_METRIC = ["pearson", "kendall", "spearman"]
    OTHER_DISTANCE_METRIC = ["Lknorm"]
    
    # Sanity checks
    check_type(x=a,allowed=TYPES_ALLOWED,name="a")
    check_type(x=b,allowed=TYPES_ALLOWED,name="b")
    check_str(x=metric,
              allowed=OTHER_DISTANCE_METRIC+SCIKIT_DISTANCE_METRIC+CORRWITH_SIMILARITY_METRIC,
              name="metric")

    # Convert types
    if isinstance(a, pd.Series):
        a = a.to_frame().T
    if isinstance(b, pd.Series):
        b = b.to_frame().T        

    # Compute similarities
    if metric in SCIKIT_DISTANCE_METRIC+OTHER_DISTANCE_METRIC:
        
        # Compute distances
        distances = compute_distance(a=a,
                                     b=b,
                                     metric=metric,
                                     **kwargs)
        
        if isin(a=metric,b=["cosine","correlation"]):            
            # Cosine distance is defined as 1.0 minus the cosine similarity.
            # Same with correlation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html)
            similarities = 1 - distances
            
            # Cosine/correlation similarity is defined on [-1,+1], which we enforce due to numerical stability
            similarities.clip(lower=-1,
                              upper=+1,
                              inplace=True)
            
            if metric=="cosine":
                if convert_to_angular_similarity:
                    # Convert cosine similarity to angular similarity (https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity)
                    similarities.iloc[:,:] = 1 - (np.arccos(similarities) / np.pi)

        else:
            similarities = -distances
        
    elif metric in CORRWITH_SIMILARITY_METRIC:
        # Pre-allocate
        similarities = []

        for c in b.index:
            
            # Get temporary score
            similarity_temp = a.corrwith(other=b.loc[c],
                                    axis=1,
                                    method=metric)
            
            # Assign name
            similarity_temp .name = c
            
            # Append
            similarities.append(similarity_temp )

        # Concatenate all scores
        similarities = pd.concat(objs=similarities, axis=1)
        
    return similarities

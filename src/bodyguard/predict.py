#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import sys, time
import numpy as np
import scipy as sp
import pandas as pd
import numbers
from functools import reduce
import itertools
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_float_dtype
  
# User
from .exceptions import WrongInputException
from .sanity_check import check_type, check_str
#------------------------------------------------------------------------------
# Cross validation tools
#------------------------------------------------------------------------------
def oos_prediction_evaluation(y_test, y_train, y_hat):
    ALLOWED=(pd.Series, np.ndarray)
    
    check_type(x=y_test, allowed=ALLOWED, name="y_test")
    check_type(x=y_train, allowed=ALLOWED, name="y_train")
    check_type(x=y_hat, allowed=ALLOWED, name="y_hat")
    
    # Total sum of squares
    SST = ((y_test - y_train.mean())**2).sum()
    
    # Residual sum of squares
    SSR = ((y_test - y_hat)**2).sum()
    
    # Out-of-sample R2
    OoSR2 = 1 - SSR/SST
        
    # (Root) Mean-squared error
    MSE = ((y_test - y_hat)**2).mean()
    RMSE = np.sqrt(MSE)
    
    # Mean absolute error        
    MAE = (y_test - y_hat).abs().mean()
    
    results = {"OoSR2" : OoSR2,
               "MSE" : MSE,
               "RMSE" : RMSE,
               "MAE" : MAE}
    
    return results
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:20:06 2022

@author: muhlbach
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .tools import print2

class PCATransformer(object):
    """
    PCA transformer but is mean-rank consistent
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 with_mean=True,
                 with_std=True,
                 n_components=1,
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto',
                 random_state=1991,
                 verbose=False):
        self.with_mean = with_mean        
        self.with_std = with_std
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.verbose = verbose
        
        # ---------------------------------------------------------------------
        # Instantiate
        # ---------------------------------------------------------------------
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.pca = PCA(n_components=self.n_components,
                       copy=self.copy,
                       whiten=self.whiten,
                       svd_solver=self.svd_solver,
                       tol=self.tol,
                       iterated_power=self.iterated_power,
                       random_state=self.random_state)
        
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------        
    def fit(self,X):
        
        # Copy
        X_standardized = X.copy()

        # Standardize
        X_standardized[:] = self.scaler.fit_transform(X=X)
        
        # Fit PCA
        self.pca.fit(X=X_standardized)
        
        return self
    
    def transform(self, X):
        
        # Copy
        X_standardized = X.copy()
        
        # Standardize
        X_standardized[:] = self.scaler.transform(X=X)
        X_mean = X_standardized.mean(axis=1)
        
        # Run mean
        X_pca = self.pca.transform(X=X_standardized)
        
        # If we reduce dimensionality all the way down to 1 dimension, we check sign on covariance matrix and potentially flip
        if self.n_components==1:
            # Estimate covariance between principal component and original mean
            cov_matrix_pca_mean = np.cov(m=X_pca.reshape(-1,), y=X_mean)
            
            # Extract covariance
            cov_pca_mean = cov_matrix_pca_mean[1,0]
            
            if np.sign(cov_pca_mean)==-1:
                if self.verbose:
                    print2(f"Flipping first principal components as covariance with standardized mean was less than one (={cov_pca_mean})")
                
                # Negate Y if signs are different
                X_pca = -X_pca
                
        if isinstance(X,pd.DataFrame):
            X_pca = pd.DataFrame(data=X_pca,
                                 index=X.index)
            
            X_pca.columns = [f"PCA{i}" for i in range(1,X_pca.shape[1]+1)]
                

        return X_pca
            
    def fit_transform(self,X):
        
        # Fit
        self.fit(X=X)
        
        # Transform
        X_pca = self.transform(X=X)
        
        
        return X_pca
            
        
            
            
            
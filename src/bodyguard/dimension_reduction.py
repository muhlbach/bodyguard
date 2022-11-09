#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:20:06 2022

@author: muhlbach
"""
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from .tools import (print2, generate_grid_from_dict)
from .sanity_check import check_str


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
class DimensionReducer(object):
    """
    Reduce dimensions of your data, either via PCA or T-SNE
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 # Common parameters
                 method="tsne",
                 tune_hyperparameters=True,
                 retune_always=False,
                 n_components=2,
                 initial_pca_thres=50,                 
                 verbose=False,
                 random_state=1991,
                 flip_sign_if_possible=True,
                 
                 # Standardization
                 with_mean=True,
                 with_std=True,

                 # PCA parameters
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto',

                # T-SNE defaults
                
                    
                perplexity_default=30,
                early_exaggeration_default=12,
                learning_rate_default=200,
                angle_default=0.5,

                 
                 # T-SNE (tuning)parameters
                 perplexity=[5,10,20,30,40,50],
                 learning_rate=[10,50,100,200,500,750,1000],
                 early_exaggeration=[10.0,12.0,14.0],                 
                 angle=[0.3,0.5,0.7],
                 
                 tsne_method='barnes_hut',
                 metric='euclidean',
                 init='pca',
                 min_grad_norm=1e-07,
                 n_iter=100000, #100000
                 n_iter_without_progress=1000,
                 n_jobs=-1
                 ):
        self.method = method
        self.tune_hyperparameters = tune_hyperparameters
        self.retune_always = retune_always
        self.n_components = n_components
        self.initial_pca_thres = initial_pca_thres        
        self.random_state = random_state
        self.verbose = verbose        
        self.flip_sign_if_possible = flip_sign_if_possible
        self.with_mean = with_mean        
        self.with_std = with_std

        self.perplexity_default = perplexity_default
        self.early_exaggeration_default = early_exaggeration_default
        self.learning_rate_default = learning_rate_default
        self.angle_default = angle_default

        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.min_grad_norm = min_grad_norm
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.early_exaggeration = early_exaggeration
        self.tsne_method = tsne_method
        self.metric = metric
        self.angle = angle
        self.init = init
        self.n_jobs = n_jobs


        # ---------------------------------------------------------------------
        # Sanity check
        # ---------------------------------------------------------------------
        check_str(x=self.method,
                  allowed=self.METHOD_ALLOWED,
                  name="method")

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
        
    # ---------------------------------------------------------------------
    # Class variables
    # ---------------------------------------------------------------------
    METHOD_ALLOWED = ["pca", "tsne", "PCA", "TSNE"]    
        
    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
    def _tune_tsne(self, X):
                        
        # Form params and grid
        params = {"perplexity" : self.perplexity,
                  "learning_rate" : self.learning_rate,
                  "angle" : self.angle,
                  "early_exaggeration" : self.early_exaggeration
                  }

        # Turn into grid
        params_grid = generate_grid_from_dict(d=params)

        # Pre-allocate performances
        self.performance = []

        # Run through all        
        for cnt,par in enumerate(params_grid):
            
            if self.verbose:
                print(f"\n Running with {par} ~ {cnt+1}/{len(params_grid)}")

            # Initialize T-SNE
            tsne = TSNE(n_components=self.n_components,
                        perplexity=par["perplexity"],
                        early_exaggeration=par["early_exaggeration"],
                        learning_rate=par["learning_rate"],
                        angle=par["angle"],
                        metric=self.metric,
                        init=self.init, 
                        random_state=self.random_state,
                        method=self.tsne_method,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose,
                        min_grad_norm=self.min_grad_norm,
                        n_iter=self.n_iter,
                        n_iter_without_progress=self.n_iter_without_progress)
        
            # Construct components
            tsne.fit(X)
            
            self.performance.append(tsne.kl_divergence_)
            
            if self.verbose:
                print2(f"""Current KL: {tsne.kl_divergence_} \nOptimal KL: {min(self.performance)}""")
    
        # At this point, we gave found the optimal parameters
        optim_index = np.argmin(self.performance)
        
        # Extract optimal parameters
        optim_par = params_grid[optim_index]
        
        for par_key,par_value in optim_par.items():
            # Set attribute 
            setattr(self, par_key+"_optim", par_value)
            
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------        
    def fit(self,X):
        
        # Copy
        X_standardized = X.copy()
        
        # Obtain columns
        self.N_col = X.shape[1]
        
        # Standardize
        X_standardized[:] = self.scaler.fit_transform(X=X)
        
        if self.method.lower()=="pca":
            # Fit PCA with given parameters
            self.pca.fit(X=X_standardized)
            
        elif self.method.lower()=="tsne":
            if self.N_col>self.initial_pca_thres:
                # Run initial PCA
                self.initial_pca = clone(self.pca)

                # Change number of components                
                self.initial_pca.set_params(**{"n_components":self.initial_pca_thres})
                
                # Fit PCA with another number of components used before T-SNE
                X_standardized = self.initial_pca.fit_transform(X=X_standardized)
            

            if self.tune_hyperparameters:           
                # Tune T-SNE
                if self.retune_always:    
                    if self.verbose:
                        print("Finetuning T-SNE")
                    
                    # Re-tune parameters
                    self._tune_tsne(X=X_standardized)
                else:
                    if self.verbose:
                        print("Re-using finedtuned T-SNE parameters")
                    
                    # Use fine-tuned parameters
                    if not all(hasattr(self, par+"_optim") for par in ["perplexity", "learning_rate", "angle", "early_exaggeration"]):
                        if self.verbose:
                            print("Not all finetuned parameters are attributes. Finetuning T-SNE.")
                        
                        self._tune_tsne(X=X_standardized)
                        
            else:
                if self.verbose:
                    print("Use default T-SNE hyperparameters")
                # Use fallback/default options
                for par in ["perplexity", "learning_rate", "angle", "early_exaggeration"]:
                    par_default = getattr(self, par+"_default")
                    setattr(self, par+"_optim", par_default)
                
            # Initial with optimal parameters
            self.tsne = TSNE(n_components=self.n_components,
                             perplexity=self.perplexity_optim,
                             early_exaggeration=self.early_exaggeration_optim,
                             learning_rate=self.learning_rate_optim,
                             angle=self.angle_optim,
                             metric=self.metric,
                             init=self.init, 
                             random_state=self.random_state,
                             method=self.tsne_method,
                             n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             min_grad_norm=self.min_grad_norm,
                             n_iter=self.n_iter,
                             n_iter_without_progress=self.n_iter_without_progress)
                
        return self

    def transform(self, X):
        
        # Copy
        X_standardized = X.copy()
        
        # Standardize
        X_standardized[:] = self.scaler.transform(X=X)
        
        if self.method.lower()=="pca":
            X_transformed = self.pca.transform(X=X_standardized)
        
        elif self.method.lower()=="tsne":
            if self.N_col>self.initial_pca_thres:
                # Fit PCA with given parameters
                X_standardized = self.initial_pca.transform(X=X_standardized)

            # Note: We need fit + transform as T-SNE does not support transform only            
            X_transformed = self.tsne.fit_transform(X=X_standardized)                

        if self.flip_sign_if_possible:
            # If we reduce dimensionality all the way down to 1 dimension, we check sign on covariance matrix and potentially flip
            if self.n_components==1:
                
                # Compute mean which is used to check if directions agree
                X_mean = X_standardized.mean(axis=1)
    
                # Estimate covariance between component and original mean
                cov_matrix = np.cov(m=X_transformed.reshape(-1,), y=X_mean)
                
                # Extract covariance
                cov_scalar = cov_matrix[1,0]
                
                if np.sign(cov_scalar)==-1:
                    if self.verbose:
                        print2(f"Flipping first and only component as covariance with standardized mean was negative (={cov_scalar})")
                    
                    # Negate
                    X_transformed = -X_transformed
                
        if isinstance(X,pd.DataFrame):
            X_transformed = pd.DataFrame(data=X_transformed,
                                         index=X.index)
            
            X_transformed.columns = [f"{self.method}{i}" for i in range(1,X_transformed.shape[1]+1)]
                

        return X_transformed


    def fit_transform(self,X):
        
        # Fit
        self.fit(X=X)
        
        # Transform
        X_transformed = self.transform(X=X)
        
        
        return X_transformed
























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
            
        
            
            
            
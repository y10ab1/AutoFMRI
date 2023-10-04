import torch
import random
import numpy as np
import shap
import os
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from utils.patchify import patchify_by_cube, get_top_k_patches, patchify_by_atlas

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from copy import deepcopy

from nilearn.image import new_img_like

from scipy.stats import uniform

from data_loader.data_loaders import HaxbyDataLoader
from data_loader.VKMC import VKMCDataLoader

from utils.model import get_model
from utils.helper import calculate_shap_values



class AutoEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, data_dir=None, subject=None, stage1_model=None, stage2_model=None, stage1_model_type=None, stage2_model_type=None, kfold=4, atlas_name=None, 
                 topk_patches=0.2, topk_percent_shap=0.01, max_patches_ratio=0.5, n_jobs=1, result_dir='results', ref_img=None,
                 random_search=True, sample_to_calculate_shap=1.0):
        self.data_dir = data_dir
        self.subject = subject
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage1_model_type = stage1_model_type
        self.stage2_model_type = stage2_model_type
        self.kfold = kfold
        self.atlas_name = atlas_name
        self.topk_patches = topk_patches
        self.topk_percent_shap = topk_percent_shap
        self.n_jobs = n_jobs
        self.result_dir = result_dir
        self.max_patches_ratio = max_patches_ratio
        self.ref_img = ref_img
        self.random_search = random_search
        self.sample_to_calculate_shap = sample_to_calculate_shap # sample k% of the training data to calculate SHAP values (saving time), default: 100%
        
        assert self.ref_img is not None, 'ref_img needs to be specified!'

    def fit(self, X, y):
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        self.patch_masks = patchify_by_atlas(X, self.ref_img, atlas_name=self.atlas_name)


        patch_scores = []
        
        # calculate patch_scores
        patch_scores = [cross_val_score(estimator=deepcopy(self.stage1_model),
                                        X=X[:, patch_mask[0], patch_mask[1], patch_mask[2]],
                                        y=y,
                                        cv=self.kfold,
                                        n_jobs=self.n_jobs,
                                        ).mean()
                        for patch_mask in self.patch_masks]

        # Adding a very small unique value to each score
        epsilon = 1e-10
        patch_scores = [score + i*epsilon for i, score in enumerate(patch_scores)]
        assert len(patch_scores) == len(set(patch_scores)), 'There are duplicate patch scores!'
        
        patch_scores, self.patch_masks = zip(*sorted(zip(patch_scores, self.patch_masks), reverse=True))
        
        print('Sorted patch scores:', patch_scores)
        
        # Limit the number of patches to be considered (default: 50% of the total number of patches)
        self.patch_masks = self.patch_masks[:int(self.max_patches_ratio * len(self.patch_masks))]
            
        # calculate SHAP values for each patch
        # patch_shap_values = []
        patch_abs_shap_values = []
        print(f'Calculating SHAP values for {len(self.patch_masks)} patches...')
        for patch_mask in tqdm(self.patch_masks):
            # get the data with the current patch mask 
            X_train_patch = X[:, patch_mask[0], patch_mask[1], patch_mask[2]]
            
            # fit the model on the current patch
            model = deepcopy(self.stage1_model)
            model.fit(X_train_patch, y)
            
            # Sample 10% of the training data to calculate SHAP values (saving time)
            sampled_idx = np.random.choice(len(X_train_patch), size=int(self.sample_to_calculate_shap * len(X_train_patch)), replace=False)
            sampled_X_train_patch = X_train_patch[sampled_idx]         
            
            # calculate SHAP values
            shap_values = calculate_shap_values(model, sampled_X_train_patch, self.stage1_model_type)
            mean_shap_values = np.mean(shap_values, axis=(0, 1)) # (n_samples, n_voxels), get contribution of each voxel and retain the direction(+/-)
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 1)) # (n_samples, n_voxels), get contribution of each voxel and get the magnitude
            print('Shape of mean_abs_shap_values:', mean_abs_shap_values.shape)
            # patch_shap_values.append(mean_shap_values)
            patch_abs_shap_values.append(mean_abs_shap_values)
            
            # delete variables that are no longer needed
            del X_train_patch
            del model



        stage2_estimator = self.Stage2Estimator(patch_masks = self.patch_masks,
                                                patch_shap_values = patch_abs_shap_values,
                                                topk_patches = self.topk_patches,
                                                topk_percent_shap = self.topk_percent_shap,
                                                stage2_model = self.stage2_model,
                                                result_dir = self.result_dir)
        if self.random_search:

            print('Start RandomizedSearchCV...')
            params = {
                'topk_patches': list(range(1, len(self.patch_masks))), # len(self.patch_masks) is the maximum number of patches
                'topk_percent_shap': uniform(0.01, 0.5),
            }
            
            # Define the GA search
            scorer = make_scorer(accuracy_score)
            ra_search = RandomizedSearchCV(estimator=stage2_estimator,
                                            param_distributions=params,
                                            scoring=scorer,
                                            cv=StratifiedKFold(n_splits=self.kfold, shuffle=True),
                                            n_jobs=self.n_jobs,
                                            n_iter=2*len(self.patch_masks)+10, # add 10 more iterations if the number of patches is small
                                            verbose=1,
            )
            
            ra_search.fit(X, y)
            print('Best params:', ra_search.best_params_)
            print('Best score:', ra_search.best_score_)
            print('Done fitting RandomizedSearchCV...')
            self.high_performance_voxels_mask = ra_search.best_estimator_.high_performance_voxels_mask
            self.best_model = ra_search.best_estimator_
        else:
            # Fit the stage 2 estimator
            print('Start fitting stage2_estimator...')
            stage2_estimator.fit(X, y)
            print('Done fitting stage2_estimator...')
            self.high_performance_voxels_mask = stage2_estimator.high_performance_voxels_mask
            self.best_model = stage2_estimator


        
        
        img_high_performance_voxels_mask = np.zeros_like(self.ref_img.get_fdata())
        img_high_performance_voxels_mask[self.high_performance_voxels_mask[0],
                                        self.high_performance_voxels_mask[1],
                                        self.high_performance_voxels_mask[2]] = 1
        self.img_high_performance_voxels_mask = new_img_like(ref_niimg=self.ref_img,
                                                                data=img_high_performance_voxels_mask)
                                                            
        
        
    def predict(self, X):
        y_pred = self.best_model.predict(X)
        y_pred = self.le.inverse_transform(y_pred)
        return y_pred

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    
    
    def fit_model_on_patch(self, model, X_train_patch, y_train):
        new_model = deepcopy(model)
        new_model.fit(X_train_patch, y_train)
        return new_model
    
    class Stage2Estimator(BaseEstimator, ClassifierMixin):
        def __init__(self, patch_masks, patch_shap_values, topk_patches, topk_percent_shap, stage2_model, result_dir):
            self.patch_masks = patch_masks # (n_patches, x_indices, y_indices, z_indices)
            self.patch_shap_values = patch_shap_values # (n_patches, n_features)
            self.topk_patches = topk_patches if topk_patches >= 1 else int(topk_patches * len(patch_masks))
            self.topk_percent_shap = topk_percent_shap
            self.stage2_model = stage2_model
            self.result_dir = result_dir

        def fit(self, X, y=None):
            
            # Collect voxels with top k percent SHAP values from each top k performance patches
            
            # Initialize lists to store the x, y, and z indices
            x_indices = []
            y_indices = []
            z_indices = []

            for i, patch_mask in enumerate(self.patch_masks[:self.topk_patches]):
                top_k_shap_values_idx = np.argsort(-self.patch_shap_values[i])[:int(self.topk_percent_shap * len(self.patch_shap_values[i]))]
                top_k_shap_values_idx = np.unravel_index(top_k_shap_values_idx, patch_mask[0].shape)
                
                # get original indices in X array
                original_x_indices = patch_mask[0][top_k_shap_values_idx]
                original_y_indices = patch_mask[1][top_k_shap_values_idx]
                original_z_indices = patch_mask[2][top_k_shap_values_idx]
                
                # append to the respective lists
                x_indices.append(original_x_indices)
                y_indices.append(original_y_indices)
                z_indices.append(original_z_indices)

            # flatten and convert the lists to numpy arrays
            x_indices = np.concatenate(x_indices)
            y_indices = np.concatenate(y_indices)
            z_indices = np.concatenate(z_indices)
            
            high_performance_voxels_mask = np.array([x_indices, y_indices, z_indices])
            
            X_train = X[:, x_indices, y_indices, z_indices]
            
            model = deepcopy(self.stage2_model)
            model.fit(X_train, y)
            
            self.high_performance_voxels_mask = high_performance_voxels_mask
            self.model = model

            return self
        
        def predict(self, X):
            x_indices, y_indices, z_indices = self.high_performance_voxels_mask
            X = X[:, x_indices, y_indices, z_indices]
            return self.model.predict(X)
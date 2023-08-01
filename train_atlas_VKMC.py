import torch
import skorch
import sklearn
import random
import pandas as pd
import argparse
import numpy as np
import shap
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn.base import BaseEstimator, ClassifierMixin
from utils.patchify import patchify_by_cube, get_top_k_patches, patchify_by_atlas
from data_loader.data_loaders import HaxbyDataLoader
from data_loader.VKMC import VKMCDataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from cuml.ensemble import RandomForestClassifier as cuRF
from model.model_gelu_cnn import CNN
from skorch import NeuralNetClassifier

from tqdm import tqdm
from copy import deepcopy

from nilearn.image import new_img_like

from utils.model import get_model
from utils.config import get_args

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from scipy.stats import uniform


# Fix random seed for reproducibility
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
same_seeds(123)

class MyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, data_dir=None, subject=None, stage1_model=None, stage2_model=None, stage1_model_type=None, stage2_model_type=None, kfold=5, atlas_name=None, 
                 topk_patches=0.2, topk_percent_shap=0.2, max_patches_ratio=0.5, n_jobs=1, result_dir='results'):
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

    def fit(self, X, y):
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        self.patch_masks = patchify_by_atlas(X, fMRIDataLoader.reference_img, atlas_name=self.atlas_name)


        patch_scores = []
        
        # calculate patch_scores
        patch_scores = [cross_val_score(estimator=deepcopy(self.stage1_model),
                                        X=X[:, patch_mask[0], patch_mask[1], patch_mask[2]],
                                        y=y,
                                        cv=self.kfold,
                                        n_jobs=4).mean()
                        for patch_mask in self.patch_masks]

        # Adding a very small unique value to each score
        epsilon = 1e-10
        patch_scores = [score + i*epsilon for i, score in enumerate(patch_scores)]
        
        
        print('Every score is unique:', len(patch_scores) == len(set(patch_scores)))
        patch_scores, self.patch_masks = zip(*sorted(zip(patch_scores, self.patch_masks), reverse=True))
        
        print('Patch scores:', patch_scores)
        
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
            sampled_idx = np.random.choice(len(X_train_patch), size=int(0.1 * len(X_train_patch)), replace=False)
            sampled_X_train_patch = X_train_patch[sampled_idx]          
            # sampled_X_train_patch = X_train_patch        
            # sampled_X_train_patch = shap.kmeans(X_train_patch, int(0.1 * len(X_train_patch))) # use kmeans to sample 10% of the training data
            
            # calculate SHAP values
            shap_values = self.calculate_shap_values(model, sampled_X_train_patch, self.stage1_model_type)
            mean_shap_values = np.mean(shap_values, axis=(0, 1)) # (n_samples, n_voxels), get contribution of each voxel and retain the direction(+/-)
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 1)) # (n_samples, n_voxels), get contribution of each voxel and get the magnitude
            print('Shape of mean_abs_shap_values:', mean_abs_shap_values.shape)
            # patch_shap_values.append(mean_shap_values)
            patch_abs_shap_values.append(mean_abs_shap_values)
            
            # delete variables that are no longer needed
            del X_train_patch
            del model
                

        # sort patch_masks and corresponding patch_abs_shap_values by patch_scores, from high to low
        # patch_scores, self.patch_masks, patch_abs_shap_values = zip(*sorted(zip(patch_scores, self.patch_masks, patch_abs_shap_values), reverse=True))
        
        # print('Patch scores:', patch_scores)
        # print('Shape of patch_abs_shap_values:', np.array(patch_abs_shap_values).shape)
        # print('Shape of patch_masks:', np.array(self.patch_masks).shape)
         



        stage2_estimator = self.Stage2Estimator(patch_masks = self.patch_masks,
                                                patch_shap_values = patch_abs_shap_values,
                                                topk_patches = self.topk_patches,
                                                topk_percent_shap = self.topk_percent_shap,
                                                stage2_model = self.stage2_model,
                                                result_dir = self.result_dir)
        
        print('Start RandomizedSearchCV...')
        params = {
            'topk_patches': list(range(1, len(self.patch_masks))), # len(self.patch_masks) is the maximum number of patches
            'topk_percent_shap': uniform(0.01, 0.2),
        }
        
        # Define the GA search
        scorer = make_scorer(accuracy_score)
        ra_search = RandomizedSearchCV(estimator=stage2_estimator,
                                        param_distributions=params,
                                        scoring=scorer,
                                        cv=self.kfold,
                                        # n_jobs=self.n_jobs,
                                        n_jobs=4,
                                        n_iter=2*len(self.patch_masks)+10, # add 10 more iterations if the number of patches is small
                                        verbose=1,
        )
        
        ra_search.fit(X, y)
        print('Best params:', ra_search.best_params_)
        print('Best score:', ra_search.best_score_)
        print('Done fitting RandomizedSearchCV...')


        self.high_performance_voxels_mask = ra_search.best_estimator_.high_performance_voxels_mask
        self.best_model = ra_search.best_estimator_
        
        img_high_performance_voxels_mask = np.zeros_like(fMRIDataLoader.reference_img.get_fdata())
        img_high_performance_voxels_mask[self.high_performance_voxels_mask[0],
                                        self.high_performance_voxels_mask[1],
                                        self.high_performance_voxels_mask[2]] = 1
        self.img_high_performance_voxels_mask = new_img_like(ref_niimg=fMRIDataLoader.reference_img,
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



    def calculate_shap_values(self, model, X_train_patch, model_type):
        
        # # sample 10% of the training data
        # sampled_idx = np.random.choice(len(X_train_patch), size=int(0.1 * len(X_train_patch)), replace=False)
        # X_train_patch = X_train_patch[sampled_idx]
        
        if model_type == 'cnn':
            X_train_patch = torch.from_numpy(X_train_patch).to('cpu')
            explainer = shap.DeepExplainer(model.module_.to('cpu'), X_train_patch)
            shap_values = explainer.shap_values(X_train_patch)

        elif model_type == 'rf':
            # explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
            explainer = shap.TreeExplainer(model)
            # shap_values = explainer.shap_values(X_train_patch)
            shap_values = explainer.shap_values(X_train_patch, check_additivity=False, approximate=True)

        else:  # model_type is not 'cnn' or 'rf', use KernelExplainer
            explainer = shap.KernelExplainer(model.predict, X_train_patch)
            shap_values = explainer.shap_values(X_train_patch)

        return shap_values
    
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
        
    
    
    
    
if __name__ == '__main__':
    
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    
    # fMRIDataLoader = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject)
    fMRIDataLoader = VKMCDataLoader(
                                    # data_dir='data/2021_Reconstruction_dataset',
                                    data_dir='data/datas/LSA_all',
                                    subject=args.subject,
    )
    # X, y = fMRIDataLoader.load_volume_data()
    X, y = fMRIDataLoader.load_data()
    
    # remove label 'rest'
    X = X[y != 'rest']
    y = y[y != 'rest']
    
    # group midi numbers 62~86 into 5 classes [low, low-mid, mid, mid-high, high]
    y = np.array([float(midi) for midi in y])
    y = np.where(y <= 66, 0, y)
    y = np.where((y > 66) & (y <= 71), 1, y)
    y = np.where((y > 71) & (y <= 76), 2, y)
    y = np.where((y > 76) & (y <= 81), 3, y)
    y = np.where(y > 81, 4, y)
    
    print('Count of each class:', np.unique(y, return_counts=True))


    
    
    
    results_df = pd.DataFrame(columns=['Subject', 'Fold', 'Accuracy', 'Confusion matrix', 'Classification report'])
    total_y = []
    total_y_pred = []

    
    


    stratified_kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    for idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y), 1):
    
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        

        # Define the estimator without specific hyperparameters
        clf = MyEstimator(  data_dir=args.data_dir,
                                subject=args.subject,
                                stage1_model=get_model(args.stage1_model, args),
                                stage2_model=get_model(args.stage2_model, args),
                                stage1_model_type=args.stage1_model,
                                stage2_model_type=args.stage2_model,
                                kfold=args.kfold,
                                atlas_name=args.atlas_name,
                                topk_patches=args.topk_patches,
                                topk_percent_shap=args.topk_percent_shap,
                                n_jobs=args.n_jobs,
                                max_patches_ratio=args.max_patches_ratio,
                                result_dir=args.result_dir)
        
        print('X_train.shape:', X_train.shape, 'y_train.shape:', y_train.shape)
        print('X_test.shape:', X_test.shape, 'y_test.shape:', y_test.shape)
        clf.fit(X_train, y_train)
    
        
        # inverse transform the labels
        y_pred = clf.predict(X_test)
        
        # Save high performance voxels mask as a 3D image
        
        clf.img_high_performance_voxels_mask.to_filename(os.path.join(args.result_dir, f'high_performance_voxels_mask_fold_{idx}.nii'))

        
        # save the predictions and targets for each fold
        total_y.extend(y_test)
        total_y_pred.extend(y_pred)
        
        # evaluate the performance of stage 2 model and save the results
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Confusion matrix:', confusion_matrix(y_test, y_pred))
        print('Classification report:')
        print(classification_report(y_test, y_pred, zero_division=0))
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # save the evaluation results for each fold
        results_df = pd.concat([results_df, pd.DataFrame({'Subject': [args.subject],
                                                            'Fold': [idx],
                                                            'Accuracy': [accuracy_score(y_test, y_pred)],
                                                            'Confusion matrix': [confusion_matrix(y_test, y_pred)],
                                                            'Classification report': [report]})])
    
    # Create classification report from total confusion matrix
    total_confusion_matrix = results_df['Confusion matrix'].sum()
    total_confusion_matrix = confusion_matrix(total_y, total_y_pred)
    
            
    # Calculate total classification report
    total_classification_report = classification_report(total_y, total_y_pred, output_dict=True, zero_division=0)
    

    # print the average classification report
    print('Classification report for all K folds:')
    print(classification_report(total_y, total_y_pred, zero_division=0))
    
        
        
    # average the evaluation results for all folds
    results_df = pd.concat([results_df, pd.DataFrame({'Subject': args.subject,
                                                        'Fold': ['Total'],
                                                        'Accuracy': [results_df['Accuracy'].mean()],
                                                        'Confusion matrix': [results_df['Confusion matrix'].sum()],
                                                        'Classification report': [total_classification_report]})])
   
    
    # save the evaluation results for all predictions
    results_df.to_json(os.path.join(args.result_dir, 'results.json'), orient='records', indent=2)
    
    

    # plot and save confusion matrix as percentage
    cm = results_df['Confusion matrix'].iloc[-1] 
    ConfusionMatrixDisplay(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 
                           display_labels=np.unique(total_y), # class names
                           ).plot(cmap='Blues', values_format='.2f', xticks_rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'confusion_matrix.png')) 

    # save args as json file to result directory
    with open(os.path.join(args.result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        

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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.model_selection import cross_val_score
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
                 topk_patches=0.2, topk_percent_shap=0.2, n_jobs=1, result_dir='results'):
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

    def fit(self, X, y):
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        self.patch_masks = patchify_by_atlas(X, fMRIDataLoader.reference_img, atlas_name=self.atlas_name)


        patch_scores = []
        
        for patch_mask in self.patch_masks:
            # get the data with the current patch mask 
            X_train_patch = X[:, patch_mask[0], patch_mask[1], patch_mask[2]] # (n_samples, n_voxels)
            
            # create a new unfit model for each patch from the same stage 1 model
            model = deepcopy(self.stage1_model)
            
            patch_scores.append(cross_val_score(estimator=model, 
                                                X=X_train_patch,
                                                y=y_train,
                                                cv=self.kfold,
                                                n_jobs=self.n_jobs).mean())
        
        # get top k performance patches and save the selected patch masks as a 3D image for visualization
        selected_patch_masks, high_performance_voxels_mask = get_top_k_patches(patch_scores = patch_scores,
                                                                               patch_masks = self.patch_masks,
                                                                               X_train = X,
                                                                               topk_patches = len(patch_scores)*self.topk_patches if self.topk_patches < 1 else self.topk_patches,
                                                                               ref_niimg = fMRIDataLoader.reference_img,
                                                                               output_filename = os.path.join(self.result_dir, f'selected_patch_masks_fold_{idx+1}.nii'))
        
        
        # retrains the stage 1 model on the selected patches
        high_performance_voxels_mask = self.run_model_on_patches(selected_patch_masks = selected_patch_masks,
                                                                high_performance_voxels_mask = high_performance_voxels_mask,
                                                                X_train = X,
                                                                y_train = y,
                                                                stage1_model = self.stage1_model)
        
        # stage 2
        high_performance_voxels_mask = high_performance_voxels_mask.astype(bool)
            
        # get the data with high performance voxels
        X_train_high_performance_voxels = X_train[:, high_performance_voxels_mask]


        # train stage 2 model on whole K-1 folds
        model = deepcopy(self.stage2_model)
        model.fit(X_train_high_performance_voxels, y)
        
        self.model = model
        self.high_performance_voxels_mask = high_performance_voxels_mask
    
        
        
        
        return self

    def predict(self, X):
        
        # apply high performance voxels mask to the test data
        X_test_high_performance_voxels = X[:, self.high_performance_voxels_mask]
        
        # predict the labels
        y_pred = self.model.predict(X_test_high_performance_voxels)
        y_pred = self.le.inverse_transform(y_pred.astype(int))
        
        return y_pred

    def score(self, X, y):
        # your scoring code
        # for example, if your scoring method is accuracy, you can use:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    
    
    def fit_model_on_patch(self, model, X_train_patch, y_train):
        new_model = deepcopy(model)
        new_model.fit(X_train_patch, y_train)
        return new_model

    def calculate_shap_values(self, model, X_train_patch, model_type):
        
        if model_type == 'cnn':
            X_train_patch = torch.from_numpy(X_train_patch).to('cpu')
            
            explainer = shap.DeepExplainer(model.module_.to('cpu'), X_train_patch)
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_train_patch)
        return shap_values

    def get_high_performance_voxels_mask(self, patch_mask, high_performance_voxels_mask, mean_abs_shap_values, top_k_shap_values_idx):
        high_performance_voxels_mask[patch_mask[0][top_k_shap_values_idx],
                                    patch_mask[1][top_k_shap_values_idx],
                                    patch_mask[2][top_k_shap_values_idx]] = np.mean(mean_abs_shap_values, axis=0)[top_k_shap_values_idx]
        return high_performance_voxels_mask

    def run_model_on_patches(self, selected_patch_masks, high_performance_voxels_mask, X_train, y_train, stage1_model):
        selected_patch_masks_loader = tqdm(selected_patch_masks)
        for patch_mask in selected_patch_masks_loader:
            X_train_patch = X_train[:, patch_mask[0], patch_mask[1], patch_mask[2]]

            model = self.fit_model_on_patch(stage1_model, X_train_patch, y_train)

            shap_values = self.calculate_shap_values(model, X_train_patch, self.stage1_model_type)
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=1)

            top_k_shap_values_idx = np.argsort(-np.mean(mean_abs_shap_values, axis=0))[:int(self.topk_percent_shap * X_train_patch.shape[1])]
            print('Number of voxels with topk_percent_shap SHAP values:', len(top_k_shap_values_idx))

            top_k_shap_values_idx = np.unravel_index(top_k_shap_values_idx, patch_mask[0].shape)
            high_performance_voxels_mask = self.get_high_performance_voxels_mask(patch_mask, high_performance_voxels_mask, mean_abs_shap_values, top_k_shap_values_idx)

            new_img_like(ref_niimg=fMRIDataLoader.reference_img,
                        data=high_performance_voxels_mask).to_filename(os.path.join(self.result_dir, f'high_performance_voxels_mask_fold_{idx+1}.nii'))

            selected_patch_masks_loader.set_description(f'Fold {idx+1}/{self.kfold}')
        
        return high_performance_voxels_mask

        
    
    
    
    
if __name__ == '__main__':
    
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    
    fMRIDataLoader = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject)
    X, y = fMRIDataLoader.load_data()
    
    stratified_kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    
    results_df = pd.DataFrame(columns=['Subject', 'Fold', 'Accuracy', 'Confusion matrix', 'Classification report'])
    total_y = []
    total_y_pred = []

    
    for idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
    
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
                            result_dir=args.result_dir)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train)
        
        # inverse transform the labels
        y_pred = clf.predict(X_test)
        
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
        results_df = pd.concat([results_df, pd.DataFrame({'Subject': args.subject,
                                                            'Fold': [idx + 1],
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
        
    # evolved_estimator = GASearchCV()
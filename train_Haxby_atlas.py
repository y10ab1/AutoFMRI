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

from model.autofmri import AutoEstimator



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
    
    
if __name__ == '__main__':
    
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    
    fMRIDataLoader = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject)
    X, y = fMRIDataLoader.load_volume_data()
    
    # train test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    
    
    
    
    
    results_df = pd.DataFrame(columns=['Subject', 'Fold', 'Accuracy', 'Confusion matrix', 'Classification report'])
    total_y = []
    total_y_pred = []

    
    


    stratified_kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    for idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y), 1):
    
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        

        # Define the estimator without specific hyperparameters
        clf = AutoEstimator(  data_dir=args.data_dir,
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
        
    # save total classification report as csv file
    pd.DataFrame(total_classification_report).transpose().to_csv(os.path.join(args.result_dir, 'total_classification_report.csv'))

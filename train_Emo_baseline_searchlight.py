import torch
import random
import pandas as pd
import numpy as np
import shap
import os
import time
import matplotlib.pyplot as plt
import json


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder


from tqdm import tqdm

from nilearn.image import new_img_like

from utils.model import get_model
from utils.config import get_args
from data_loader.Emo import EmoDataLoader

from nilearn.decoding import SearchLight
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
from nilearn.image import new_img_like, threshold_img

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
    
    
    # fMRIDataLoader = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject)
    fMRIDataLoader = EmoDataLoader(
                                    subject=args.subject,
    )
    X, y = fMRIDataLoader.load_volume_data()
    
    whole_brain_mask = compute_epi_mask(fMRIDataLoader.reference_img)

    # drop the label with 'end'
    X = X[y != 'end']
    y = y[y != 'end']
    
    print('Count of each class:', np.unique(y, return_counts=True))
    
    results_df = pd.DataFrame(columns=['Subject', 'Fold', 'Accuracy', 'Confusion matrix', 'Classification report'])
    total_y = []
    total_y_pred = []

    
    


    stratified_kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    for idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y), 1):
    
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Searchlight for feature selection
        searchlight = SearchLight(whole_brain_mask,
                                    process_mask_img=whole_brain_mask,
                                    radius=fMRIDataLoader.reference_img.header.get_zooms()[0] * 2, # 2 times the voxel size
                                    verbose=1,
                                    n_jobs=args.n_jobs,
        )
        # make a copy of the data as nibabel format
        X_train_nib = new_img_like(fMRIDataLoader.reference_img, X_train.transpose(1, 2, 3, 0))
        searchlight.fit(X_train_nib, y_train)
        # select the voxels with high scores
        high_performance_voxels_mask = searchlight.scores_ > np.percentile(searchlight.scores_, 10) # 10% percentile
        print('high_performance_voxels_mask.shape:', high_performance_voxels_mask.shape)
        X_train = X_train[:, high_performance_voxels_mask]
        X_test = X_test[:, high_performance_voxels_mask]
        
        
        

        # Define the estimator without specific hyperparameters
        clf = get_model(args.stage1_model, args)
        
        
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
    results_df = pd.concat([results_df, pd.DataFrame({'Subject': [" ".join(args.subject)],
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
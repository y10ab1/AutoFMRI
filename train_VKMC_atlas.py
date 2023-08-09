import torch
import random
import pandas as pd
import numpy as np
import shap
import os
import time
import matplotlib.pyplot as plt
import json
import scipy


from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder


from tqdm import tqdm

from nilearn.image import new_img_like

from utils.model import get_model
from utils.config import get_args
from data_loader.data_loaders import HaxbyDataLoader
from data_loader.VKMC import VKMCDataLoader


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

def post_process_labels(X, y):
    # remove label 'rest'
    X = X[y != 'rest']
    y = y[y != 'rest']
    
    # group midi numbers 62~86 into 5 classes {low, low-mid, mid, mid-high, high}
    y = np.array([float(midi) for midi in y])
    y = np.where(y <= 66, 0, y)
    y = np.where((y > 66) & (y <= 71), 1, y)
    y = np.where((y > 71) & (y <= 76), 2, y)
    y = np.where((y > 76) & (y <= 81), 3, y)
    y = np.where(y > 81, 4, y)
    
    # Create a dictionary to map the numeric labels to their string counterparts
    midi_dict = {0: 'low', 1: 'low-mid', 2: 'mid', 3: 'mid-high', 4: 'high'}

    # Apply the mapping to each element in y
    y = np.vectorize(midi_dict.get)(y)
    print('Count of each class:', np.unique(y, return_counts=True))
    
    return X, y
    
    
if __name__ == '__main__':
    
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    
    # fMRIDataLoader = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject)
    fMRIDataLoader = VKMCDataLoader(
                                    data_dir=args.data_dir,
                                    # data_dir='data/datas/LSA_all',
                                    subject=args.subject,
                                    augment=args.augment,
    )
    
    X, y = fMRIDataLoader.load_volume_data()
    X, y = post_process_labels(X, y)
    
    #################################### For dividing data into 4 folds by runs and sessions ####################################
    # create 4 folds by runs and sessions
    # X, y = [], []
    # for s in range(1, 3): # session: 1, 2
    #     for r in range(1, 9, 4): # run: 1, 5
    #         X_, y_ = fMRIDataLoader.load_volume_data(runs=list(range(r,r+4)), sessions=[s])
    #         X_, y_ = post_process_labels(X_, y_)
    #         X.append(X_)
    #         y.append(y_)
    # groups = np.concatenate([np.repeat(i, len(x)) for i, x in enumerate(X)]) # groups: 0, 1, 2, 3
    # print('Groups:', groups, 'Number of groups:', len(np.unique(groups)))
    # X = np.concatenate(X)
    # y = np.concatenate(y)
    ###########################################################################################################################
    

    
    
    results_df = pd.DataFrame(columns=['Subject', 'Fold', 'Accuracy', 'Confusion matrix', 'Classification report'])
    total_y = []
    total_y_pred = []

    
    

    # kfold = GroupKFold(n_splits=args.kfold)
    kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    
    for idx, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    # for idx, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups=groups), 1):
    
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # remove interpolated data for test set
        # Filtering the test data based on the interpolation condition
        mask = np.array([x.tobytes() not in fMRIDataLoader.interpolated_data_set for x in X_test])
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        
        # Using Scipy to interpolate the between volumes to get 1 volume per second
        # zoom_factor = (2, 1) if X_train.ndim == 2 else (2, 1, 1, 1)
        # X_train = scipy.ndimage.zoom(X_train, zoom_factor, order=1)
        # y_train = np.repeat(y_train, 2)

        
        
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
                                result_dir=args.result_dir,
                                ref_img=fMRIDataLoader.reference_img)
        
        

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
                        #    display_labels=np.unique(total_y), # class names
                           display_labels=np.array(['low', 'low-mid', 'mid', 'mid-high', 'high']), # class names
                           ).plot(cmap='Blues', values_format='.2f', xticks_rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'confusion_matrix.png')) 

    # save args as json file to result directory
    with open(os.path.join(args.result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # save total classification report as csv file
    pd.DataFrame(total_classification_report).transpose().to_csv(os.path.join(args.result_dir, 'total_classification_report.csv'))
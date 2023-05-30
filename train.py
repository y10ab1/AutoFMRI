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

from utils.patchify import patchify_by_cube, get_top_k_patches
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
SEED = 123
same_seeds(SEED)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, default='data/haxby2001/subj4/first_level_output')
    args.add_argument('--subject', type=str, default='sub-04')
    args.add_argument('--result_dir', type=str, default=f'result/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}')
    args.add_argument('--stage1_model', type=str, default='rf', help='Specify the stage 1 model', choices=['rf', 'cnn', 'cu_rf'])
    args.add_argument('--stage2_model', type=str, default='rf', help='Specify the stage 2 model', choices=['rf', 'cnn', 'cu_rf'])
    args.add_argument('--kfold', type=int, default=4, help='Specify the number of folds for cross validation')
    args.add_argument('--n_jobs', type=int, default=-1, help='Specify the number of jobs for sklearn parallel processing')
    args.add_argument('--n_bins', type=int, default=16, help='Specify the number of bins for cuml parallel processing')
    args.add_argument('--n_streams', type=int, default=8, help='Specify the number of streams for cuml parallel processing')
    args.add_argument('--topk_patches', type=int, default=5, help='Specify the number of top performance patches')
    args.add_argument('--topk_percent_shap', type=float, default=0.01, help='Specify the percentage of voxels with top SHAP values', choices=range(0, 1))
    args.add_argument('--cube_size', type=str, default="20 20 20", help='Specify the cube size for patchify in the format: "x y z" for each dimension')
    args.add_argument('--label_encoder', type=str, default=None, help='Specify the label encoder for inverse transform')
    

    # CNN hyperparameters
    args.add_argument('--num_classes', type=int, default=8, help='Specify the number of classes for classification')
    args.add_argument('--num_epochs', type=int, default=100, help='Specify the number of epochs for training')
    args.add_argument('--batch_size', type=int, default=32, help='Specify the batch size for training')
    args.add_argument('--lr', type=float, default=0.001, help='Specify the learning rate for training')
    args.add_argument('--weight_decay', type=float, default=0.0, help='Specify the weight decay for training')
    args.add_argument('--verbose', type=int, default=1, help='Specify the verbose for training')
    
    # Post processing for arguments
    parse_args = args.parse_args()
    parse_args.cube_size = tuple(map(int, parse_args.cube_size.split()))
    return parse_args

def get_model(model_name, args):
    if model_name == 'rf':
        return skRF(n_estimators=100, n_jobs=args.n_jobs)
    elif model_name == 'cu_rf':
        return cuRF(n_estimators=100, n_streams=args.n_streams, n_bins=args.n_bins)
    elif model_name == 'cnn':
        net = NeuralNetClassifier(module=CNN, 
                                    module__task='classification', 
                                    module__num_classes=args.num_classes,
                                    criterion=torch.nn.CrossEntropyLoss, 
                                    optimizer=torch.optim.Adam, 
                                    lr=args.lr,
                                    max_epochs=args.num_epochs, 
                                    batch_size=args.batch_size,
                                    train_split=None, 
                                    verbose=args.verbose,
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
        return net


def main(args):
    # get data (3D images and labels)
    le = LabelEncoder()
    fMRIDataLoader = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject)
    X, y = fMRIDataLoader.load_data()
    y = le.fit_transform(y) # encode labels to integers
    
    # get stage 1 & 2 model
    stage1_model = get_model(args.stage1_model, args)
    stage2_model = get_model(args.stage2_model, args)
    
    # split data into K folds
    stratified_kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    
    # patchify data
    patch_masks = patchify_by_cube(X, cube_size=args.cube_size, stride=args.cube_size)
    
    # create dataframes for saving results
    results_df = pd.DataFrame(columns=['Subject', 'Fold', 'Accuracy', 'Confusion matrix', 'Classification report'])
    classification_reports_df = []

    # total predictions and targets
    total_y, total_y_pred = [], []
    
    for idx, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
        patch_scores = []
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        for patch_mask in patch_masks:
            # get the data with the current patch mask 
            X_train_patch = X_train[:, patch_mask[0], patch_mask[1], patch_mask[2]] # (n_samples, n_voxels)
            
            # create a new unfit model for each patch from the same stage 1 model
            model = deepcopy(stage1_model)
            
                            

            patch_scores.append(cross_val_score(estimator=model, 
                                                X=X_train_patch,
                                                y=y_train,
                                                cv=args.kfold,
                                                n_jobs=args.n_jobs).mean())
            
        # get top k performance patches and save the selected patch masks as a 3D image for visualization
        selected_patch_masks, high_performance_voxels_mask = get_top_k_patches(patch_scores = patch_scores,
                                                                               patch_masks = patch_masks,
                                                                               X_train = X_train,
                                                                               topk_patches = args.topk_patches,
                                                                               ref_niimg = fMRIDataLoader.reference_img,
                                                                               output_filename = os.path.join(args.result_dir, f'selected_patch_masks_fold_{idx+1}.nii'))
        
        


        selected_patch_masks_loader = tqdm(selected_patch_masks)
        for patch_mask in selected_patch_masks_loader:
            X_train_patch = X_train[:, patch_mask[0], patch_mask[1], patch_mask[2]] # (n_samples, n_voxels)
            
            
            
        
            # create a new unfit model for each patch from the same stage 1 model
            model = deepcopy(stage1_model)
            model.fit(X_train_patch, y_train)
            
            # calculate SHAP values
            if args.stage1_model == 'cnn':
                # Due to some unknown reason, the shap.DeepExplainer does not work on GPU,
                # so we have to move the data to CPU though it is very slow
                
                # X_train_patch = torch.from_numpy(X_train_patch).to('cuda' if torch.cuda.is_available() else 'cpu')
                X_train_patch = torch.from_numpy(X_train_patch).to('cpu')
                
                explainer = shap.DeepExplainer(model.module_.to('cpu'), X_train_patch)
                shap_values = explainer.shap_values(X_train_patch)
                
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train_patch)
            
            # calculate the mean absolute SHAP values for each class
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=1)                
            
            # get topk_percent_shap voxels
            top_k_shap_values_idx = np.argsort(-np.mean(mean_abs_shap_values, axis=0))[:int(args.topk_percent_shap * X_train_patch.shape[1])]

            print('Number of voxels with topk_percent_shap SHAP values:', len(top_k_shap_values_idx))
            
            # map these voxels back to the original data space
            top_k_shap_values_idx = np.unravel_index(top_k_shap_values_idx, patch_mask[0].shape)
            
            # aquire a mask for these voxels
            # retain the voxels with topk_percent_shap SHAP values with its SHAP values
            # high_performance_voxels_mask[patch_mask[0][top_k_shap_values_idx], 
            #                              patch_mask[1][top_k_shap_values_idx], 
            #                              patch_mask[2][top_k_shap_values_idx]] = 1
            high_performance_voxels_mask[patch_mask[0][top_k_shap_values_idx],
                                        patch_mask[1][top_k_shap_values_idx],
                                        patch_mask[2][top_k_shap_values_idx]] = np.mean(mean_abs_shap_values, axis=0)[top_k_shap_values_idx]
                                         
            new_img_like(ref_niimg=fMRIDataLoader.reference_img,
                        data=high_performance_voxels_mask).to_filename(os.path.join(args.result_dir, f'high_performance_voxels_mask_fold_{idx+1}.nii'))
            selected_patch_masks_loader.set_description(f'Fold {idx+1}/{args.kfold}')
        
        
        high_performance_voxels_mask = high_performance_voxels_mask.astype(bool)
            
        # get the data with high performance voxels
        X_train_high_performance_voxels = X_train[:, high_performance_voxels_mask]


        # train stage 2 model on whole K-1 folds
        model = deepcopy(stage2_model)
        model.fit(X_train_high_performance_voxels, y_train)
        
        
        # test stage 2 model on the remaining fold
        X_test_high_performance_voxels = X_test[:, high_performance_voxels_mask]

        y_pred = model.predict(X_test_high_performance_voxels)

        # inverse transform the labels
        y_test = le.inverse_transform(y_test)
        y_pred = le.inverse_transform(y_pred.astype(int))
        
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
        
        
if __name__ == '__main__':
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args)
    
    # save args as json file to result directory
    with open(os.path.join(args.result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
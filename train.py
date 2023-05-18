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

from utils.patchify import patchify_by_cube
from data_loader.data_loaders import HaxbyDataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


from cuml.ensemble import RandomForestClassifier as cuRF

from model.model_gelu_cnn import CNN
from copy import deepcopy

from skorch import NeuralNetClassifier

from tqdm import tqdm

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
    args.add_argument('--result_dir', type=str, default=f'result/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}')
    args.add_argument('--subject', type=str, default='sub-04')
    args.add_argument('--stage1_model', type=str, default='rf', help='Specify the stage 1 model', choices=['rf', 'cnn', 'cu_rf'])
    args.add_argument('--stage2_model', type=str, default='rf', help='Specify the stage 2 model', choices=['rf', 'cnn', 'cu_rf'])
    args.add_argument('--kfold', type=int, default=4, help='Specify the number of folds for cross validation')
    args.add_argument('--n_jobs', type=int, default=-1, help='Specify the number of jobs for sklearn parallel processing')
    args.add_argument('--n_streams', type=int, default=8, help='Specify the number of streams for cuml parallel processing')
    args.add_argument('--k', type=int, default=5, help='Specify the number of top performance patches')
    args.add_argument('--k_shap_percent', type=float, default=0.01, help='Specify the percentage of voxels with top SHAP values', choices=range(0, 1))
    args.add_argument('--cube_size', type=tuple, default=(20, 20, 20), help='Specify the cube size for patchify')
    
    
    # CNN hyperparameters
    args.add_argument('--num_classes', type=int, default=8, help='Specify the number of classes for classification')
    args.add_argument('--num_epochs', type=int, default=100, help='Specify the number of epochs for training')
    args.add_argument('--batch_size', type=int, default=32, help='Specify the batch size for training')
    args.add_argument('--lr', type=float, default=0.001, help='Specify the learning rate for training')
    args.add_argument('--weight_decay', type=float, default=0.0, help='Specify the weight decay for training')
    args.add_argument('--verbose', type=int, default=1, help='Specify the verbose for training')
    
    
    return args.parse_args()

def get_model(model_name, args):
    if model_name == 'rf':
        return skRF(n_estimators=100, n_jobs=args.n_jobs)
    elif model_name == 'cu_rf':
        return cuRF(n_estimators=100, n_streams=args.n_streams)
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
    X, y = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject).load_data()
    
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
            
        # get top k performance patches
        print('Top k performance scores:', np.sort(patch_scores)[-args.k:])
        selected_patch_masks_idx = np.argsort(patch_scores)[-args.k:] # the indices of the top k performance patches
        selected_patch_masks = np.array(patch_masks, dtype=object)[selected_patch_masks_idx].tolist() # the top k performance patches
        high_performance_voxels_mask = np.zeros(X_train[0].shape) # the mask for high performance voxels

        
        for patch_mask in selected_patch_masks:
            X_train_patch = X_train[:, patch_mask[0], patch_mask[1], patch_mask[2]] # (n_samples, n_voxels)
            # create a new unfit model for each patch from the same stage 1 model
            model = deepcopy(stage1_model)
            model.fit(X_train_patch, y_train)
            
            # calculate SHAP values
            if args.stage1_model == 'cnn':
                explainer = shap.DeepExplainer(model.module_, X_train_patch)
            else:
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_train_patch)
            
            # calculate the mean absolute SHAP values for each class
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=1)                
            
            # get top k_shap_percent voxels
            top_k_shap_values_idx = np.argsort(-np.mean(mean_abs_shap_values, axis=0))[:int(args.k_shap_percent * X_train_patch.shape[1])]

            print('Number of voxels with top k_shap_percent SHAP values:', len(top_k_shap_values_idx))
            
            # map these voxels back to the original data space
            top_k_shap_values_idx = np.unravel_index(top_k_shap_values_idx, patch_mask[0].shape)
            
            # aquire a mask for these voxels
            high_performance_voxels_mask[patch_mask[0][top_k_shap_values_idx], 
                                         patch_mask[1][top_k_shap_values_idx], 
                                         patch_mask[2][top_k_shap_values_idx]] = 1
        
        
        high_performance_voxels_mask = high_performance_voxels_mask.astype(bool)
            
        # get the data with high performance voxels
        X_train_high_performance_voxels = X_train[:, high_performance_voxels_mask]

        # train stage 2 model on whole K-1 folds
        model = deepcopy(stage2_model)
        model.fit(X_train_high_performance_voxels, y_train)
        
        
        # test stage 2 model on the remaining fold
        X_test_high_performance_voxels = X_test[:, high_performance_voxels_mask]
        y_pred = model.predict(X_test_high_performance_voxels)
        
        # evaluate the performance of stage 2 model and save the results
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Confusion matrix:', confusion_matrix(y_test, y_pred))
        print('Classification report:', classification_report(y_test, y_pred))
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # save the evaluation results for each fold
        results_df = pd.concat([results_df, pd.DataFrame({'Subject': args.subject,
                                                            'Fold': [idx + 1],
                                                            'Accuracy': [accuracy_score(y_test, y_pred)],
                                                            'Confusion matrix': [confusion_matrix(y_test, y_pred)],
                                                            'Classification report': [report]})])
        
        # create a dataframe for the classification report and append it to the list
        report_df = pd.DataFrame(report).transpose()
        report_df['Fold'] = idx + 1
        classification_reports_df.append(report_df)
        
    # concatenate all classification report dataframes
    classification_reports_df = pd.concat(classification_reports_df, axis=0)

    # group by the index (i.e., the class labels) and calculate the mean
    average_classification_report = classification_reports_df.groupby(classification_reports_df.index).mean()

    # drop the 'Fold' column from the average classification report
    average_classification_report = average_classification_report.drop(columns='Fold')

    # print the average classification report
    print('Average classification report:')
    print(average_classification_report)
        
        
    # average the evaluation results for all folds
    results_df = pd.concat([results_df, pd.DataFrame({'Subject': args.subject,
                                                        'Fold': ['Average'],
                                                        'Accuracy': [results_df['Accuracy'].mean()],
                                                        'Confusion matrix': [results_df['Confusion matrix'].sum()],
                                                        'Classification report': [average_classification_report]})])
   
    
    # save the evaluation results for all folds
    results_df.to_csv(os.path.join(args.result_dir, 'results.csv'), index=False)
    
    # get the unique class labels from the targets
    class_labels = np.unique(y)

    # plot and save confusion matrix as percentage
    cm = results_df['Confusion matrix'].iloc[-1] 
    ConfusionMatrixDisplay(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 
                           display_labels=class_labels,
                           ).plot(cmap='Blues', values_format='.2f', xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'confusion_matrix.png'))    
        
        
if __name__ == '__main__':
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args)
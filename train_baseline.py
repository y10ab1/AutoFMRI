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

from utils.patchify import patchify_by_cube
from data_loader.data_loaders import HaxbyDataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder



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
    args.add_argument('--n_bins', type=int, default=16, help='Specify the number of bins for cuml parallel processing')
    args.add_argument('--n_streams', type=int, default=8, help='Specify the number of streams for cuml parallel processing')
    args.add_argument('--k', type=int, default=5, help='Specify the number of top performance patches')
    args.add_argument('--k_shap_percent', type=float, default=0.01, help='Specify the percentage of voxels with top SHAP values', choices=range(0, 1))
    args.add_argument('--cube_size', type=tuple, default=(20, 20, 20), help='Specify the cube size for patchify')
    args.add_argument('--label_encoder', type=str, default=None, help='Specify the label encoder for inverse transform')
    args.add_argument('--mask_file', type=str, default='data/haxby2001/subj4/mask4_vt.nii', help='Specify the mask file for masking')
    
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
    X, y = HaxbyDataLoader(data_dir=args.data_dir, subject=args.subject, mask_file=args.mask_file).load_data()
    y = le.fit_transform(y) # encode labels to integers
    X = X.squeeze() # squeeze from (n_samples, 1, 675) to (n_samples, 675)
    
    # get model
    model = get_model(args.stage1_model, args)

    # create result dataframe
    results_df = pd.DataFrame(columns=['Subject', 'Accuracy', 'Confusion matrix', 'Classification report'])
    
    
    # cross validation to get the performance of the model
    cv = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=SEED)
    scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1_macro'], n_jobs=args.n_jobs, verbose=1)
    print('Accuracy:', scores['test_accuracy'].mean())
    print('F1:', scores['test_f1_macro'].mean())
        
        
    
    # save classification report and confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=args.n_jobs, verbose=1)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    # save the evaluation results for all predictions
    results_df = pd.concat([results_df, pd.DataFrame({'Subject': args.subject,
                                                        'Accuracy': scores['test_accuracy'].mean(),
                                                        'Confusion matrix': [cm],
                                                        'Classification report': [report]})])
                                                      
    results_df.to_json(os.path.join(args.result_dir, 'results.json'), orient='records', indent=2)
    
    
    ConfusionMatrixDisplay(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 
                           display_labels=le.classes_).plot(cmap='Blues', values_format='.2f', xticks_rotation=45)    
    
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
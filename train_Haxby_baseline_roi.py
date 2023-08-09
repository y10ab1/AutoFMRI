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
from utils.config import get_args
from utils.model import get_model

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


def main(args):
    # get data (3D images and labels)
    le = LabelEncoder()
    X, y = HaxbyDataLoader(data_dir=args.data_dir, 
                           subject=args.subject, 
                           mask_file=args.mask_file).load_volume_data()
    y = le.fit_transform(y) # encode labels to integers
    X = X.squeeze() # squeeze from (n_samples, 1, 675) to (n_samples, 675)
    
    # get model
    model = get_model(args.stage1_model, args)

    # create result dataframe
    results_df = pd.DataFrame(columns=['Subject', 'Accuracy', 'Confusion matrix', 'Classification report'])
    
    # print('X shape:', X.shape)
    # print('X', X)
    # cross validation to get the performance of the model
    cv = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=SEED)
    scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1_macro'], n_jobs=args.n_jobs, verbose=1, error_score='raise')
    # scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1_macro'], verbose=1, error_score='raise')
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
    
    # save total classification report as csv file
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.result_dir, 'total_classification_report.csv'))
    
if __name__ == '__main__':
    args = get_args()
    
    # create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args)
    
    # save args as json file to result directory
    with open(os.path.join(args.result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
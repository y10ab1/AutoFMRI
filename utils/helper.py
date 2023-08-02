import numpy as np
import torch
import shap


def calculate_shap_values(self, model, X_train_patch, model_type):
    
    # # sample 10% of the training data
    # sampled_idx = np.random.choice(len(X_train_patch), size=int(0.1 * len(X_train_patch)), replace=False)
    # X_train_patch = X_train_patch[sampled_idx]
    
    if model_type == 'cnn':
        X_train_patch = torch.from_numpy(X_train_patch).to('cpu')
        explainer = shap.DeepExplainer(model.module_.to('cpu'), X_train_patch)
        shap_values = explainer.shap_values(X_train_patch)

    elif model_type in ['rf', 'rfr', 'xgb', 'xgbr']:
        # explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
        explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X_train_patch)
        shap_values = explainer.shap_values(X_train_patch, check_additivity=False)

    else:  # model_type is not 'cnn' or 'rf', use KernelExplainer
        explainer = shap.KernelExplainer(model.predict, X_train_patch)
        shap_values = explainer.shap_values(X_train_patch)

    return shap_values
from sklearn.ensemble import RandomForestClassifier as skRFC
from sklearn.ensemble import RandomForestRegressor as skRFR
from sklearn.svm import SVC as skSVC
from sklearn.svm import SVR as skSVR
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.ensemble import RandomForestRegressor as cuRFR
from skorch import NeuralNetClassifier
from model.model_gelu_cnn import CNN

import torch


def get_model(model_name, args):
    if model_name == 'rf':
        return skRFC(n_estimators=100, n_jobs=args.n_jobs)
    elif model_name == 'rfr':
        return skRFR(n_estimators=100, n_jobs=args.n_jobs)
    elif model_name == 'cu_rf':
        return cuRFC(n_estimators=100, n_streams=args.n_streams, n_bins=args.n_bins)
    elif model_name == 'cu_rfr':
        return cuRFR(n_estimators=100, n_streams=args.n_streams, n_bins=args.n_bins)
    elif model_name == 'svm':
        return skSVC()
    elif model_name == 'svr':
        return skSVR()
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
    

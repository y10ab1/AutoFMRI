from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.svm import SVC as skSVC
from cuml.ensemble import RandomForestClassifier as cuRF
from skorch import NeuralNetClassifier
from model.model_gelu_cnn import CNN



def get_model(model_name, args):
    if model_name == 'rf':
        return skRF(n_estimators=100, n_jobs=args.n_jobs)
    elif model_name == 'cu_rf':
        return cuRF(n_estimators=100, n_streams=args.n_streams, n_bins=args.n_bins)
    elif model_name == 'svm':
        return skSVC()
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
    
if __name__ == '__main__':
    # Test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='rf')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_streams', type=int, default=1)
    parser.add_argument('--n_bins', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    model = get_model(args.model_name, args)
    
    print(model)
from utils.model import get_model

if __name__ == '__main__':
    # Test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='rfr')
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
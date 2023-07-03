import argparse
import time

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
    args.add_argument('--topk_patches', type=float, default=0, help='Specify the number of top performance patches')
    args.add_argument('--topk_percent_shap', type=float, default=0.01, help='Specify the percentage of voxels with top SHAP values')
    args.add_argument('--cube_size', type=str, default="20 20 20", help='Specify the cube size for patchify in the format: "x y z" for each dimension')
    args.add_argument('--cube_stride', type=str, default="20 20 20", help='Specify the cube stride for patchify in the format: "x y z" for each dimension')
    args.add_argument('--label_encoder', type=str, default=None, help='Specify the label encoder for inverse transform')
    args.add_argument('--atlas_name', type=str, default='yeo', help='Specify the atlas name for patchify', choices=['harvard_oxford', 'juelich', 'yeo', 'yeo400', 'hcp_mmp_sym'])  

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
    parse_args.cube_stride = tuple(map(int, parse_args.cube_stride.split()))
    return parse_args
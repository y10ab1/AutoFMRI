from nilearn.image import load_img, binarize_img, index_img, clean_img, new_img_like, resample_to_img, get_data
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map
from scipy.ndimage import binary_dilation
from nilearn import datasets


import os
import numpy as np
import pandas as pd
import pandas as pd
import glob
from collections import defaultdict
from scipy import stats



class LanguageDataLoader():
    # Load data from local preprocessed nii.gz files
    def __init__(self, data_dir='data/HCP_LANGUAGE', subject=[1], mask_file=None, threshold=0.9, runs=['LR', 'RL']):
        self.data_dir = data_dir
        self.subject = [int(sub) for sub in subject] if isinstance(subject, list) else [int(subject)]
        self.runs = runs
        self.mask = load_img(mask_file) if mask_file is not None else None
        
        # resample mask if mask is too large
        if self.mask is not None:
            if self.mask.shape[0] > 100:
                self.mask = new_img_like(self.mask, self.mask.get_fdata()[::2, ::2, ::2])
                print('Resampled mask shape:', self.mask.shape)
        print('Mask shape:', self.mask.shape if self.mask is not None else None)
        
        # if mask is not made of 0s and 1s, threshold it
        if threshold is not None and self.mask is not None:
            self.mask = binarize_img(self.mask, threshold)
            
            print('Mask shape after thresholding:', self.mask.shape, 
                  '0s:', np.sum(self.mask.get_fdata() == 0),
                    '1s:', np.sum(self.mask.get_fdata() == 1))
            
        # pick the first volume of the first run as the reference image
        self.nii_files = glob.glob(os.path.join(self.data_dir, f'subj{subject[0]}/MNINonLinear/Results/tfMRI_LANGUAGE_{runs[0]}',
                                           f'tfMRI_LANGUAGE_{runs[0]}.nii.gz'))
        self.reference_img = index_img(load_img(self.nii_files[0]), 0)
        
        
    def load_data(self):
        # Each image file name is like: bottle_10.nii.gz, where bottle is the label and 10 is the run number
        
        # Load data from data_dir
        # X: 4D array of shape (n_samples, n_x, n_y, n_z)
        # y: 1D array of shape (n_samples,)
        
        pass
    
    def load_volume_data(self):
        # Load 4D data from data_dir and get labels
        # the data path should be like: AutoFMRI/data/fmri-dataset-for-emotion-recognition/Sub-01/wrsub-01_task-emotionalfaces_run-1_bold.nii the 4D data
        # the label path should be like: AutoFMRI/data/fmri-dataset-for-emotion-recognition/onsetime/task-emotionalfaces_run-1_events.tsv
        X, y = [], []
        for subject in self.subject:
            
            # the filename will look like: "data/HCP_LANGUAGE/subj1/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR.nii.gz"

            # get the run information from the nii file names eg. ['LR', 'RL']
            runs = self.runs
            # filter out the runs that are not in self.runs
            runs = [run for run in runs if run in self.runs]

            
            print('nii_files:', self.nii_files)
            print('runs:', runs)            
                
                
            fourD_data_runs = {}
            for run in runs:
                # load 4D data
                fourD_data_runs[run] = load_img(os.path.join(self.data_dir, f'subj{subject}/MNINonLinear/Results/tfMRI_LANGUAGE_{run}',
                                                    f'tfMRI_LANGUAGE_{run}.nii.gz'))
                # Masking
                if self.mask is not None:
                    self.mask = load_img(f'data/HCP_LANGUAGE/subj{subject}/mask_p0.001.nii.gz')
                    masker = NiftiMasker(mask_img=self.mask, standardize=True)
                    fourD_data_runs[run] = masker.fit_transform(fourD_data_runs[run])
                else:
                    # Just standardize the data if mask is not specified
                    fourD_data_runs[run] = clean_img(fourD_data_runs[run]).get_fdata().transpose(3, 0, 1, 2)
                    # fourD_data_runs[run] = fourD_data_runs[run].get_fdata().transpose(3, 0, 1, 2)
                    pass
                print(f'fourD_data_runs[{run}].shape:', fourD_data_runs[run].shape)
            
            TR = 0.72
            
            labels_runs = {}
            # get labels for each run
            for run in runs:
                labels = pd.read_csv(os.path.join(self.data_dir, f'subj{subject}/MNINonLinear/Results/tfMRI_LANGUAGE_{run}/EVs',
                                                   f'blocks.csv'))
                
                # columns are 'onset', 'duration','trial_type'
                # divide onset and duration by TR, so that we can get the label of each volume
                # eg. for volumes 1-30, the label is happy, for volumes 31-60, the label is sad, etc.
                labels['onset'] = labels['onset'].apply(lambda x: float(x) / TR)
                labels['duration'] = labels['duration'].apply(lambda x: float(x) / TR) 
                
                # get the label of each volume, TR = 0.72s
                # if the volume is in the range of onset and onset + duration, then the label is the trial_type
                # otherwise, the label is rest
                labels_runs[run] = ['rest'] * fourD_data_runs[run].shape[0]
                print(len(labels_runs[run]))
                for i in range(len(labels_runs[run])):
                    for j in range(len(labels)):
                        if labels['onset'][j] <= i < labels['onset'][j] + labels['duration'][j]:
                            labels_runs[run][i] = labels['trial_type'][j]
                            break
                print('label counts:', np.unique(labels_runs[run], return_counts=True))
            
            
            # volume_data = fourD_data.get_fdata()
            
            # (n_samples, n_x, n_y, n_z) if mask is None, otherwise (n_samples, n_voxels)
            X_sub = np.concatenate([fourD_data_runs[run] for run in runs], axis=0)
            y_sub = np.concatenate([labels_runs[run] for run in runs], axis=0)
            X.append(X_sub)
            y.append(y_sub)
        
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        
        # the number of volumes shoud be the same as the number of labels
        print('X.shape:', X.shape, 'y.shape:', y.shape)
        assert X.shape[0] == y.shape[0], f'The number of volumes should be the same as the number of labels, but got {X.shape[0]} volumes and {y.shape[0]} labels.'
        

        
        return X, y
        
def get_tsnr(X):
    # calculate tsnr for each voxel
    tsnr = np.mean(X, axis=0) / np.std(X, axis=0)
    print('tsnr.shape:', tsnr.shape)
    # exclude the nan/infinite voxels
    tsnr = tsnr[~np.isnan(tsnr)]
    tsnr = tsnr[~np.isinf(tsnr)]
    print('Max tsnr:', np.max(tsnr), 'Min tsnr:', np.min(tsnr) )
    print('overall tsnr:', np.mean(tsnr))
    return np.mean(tsnr)
    
if __name__ == '__main__':
    
    
    # calculate contrast for each subject
    for subj in range(1, 14):
        data_loader = LanguageDataLoader(subject=[subj])
        X, y = data_loader.load_volume_data()
        print('label counts:', np.unique(y, return_counts=True))
        _, p_values = stats.ttest_ind(X[(y == 'math') | (y == 'story')],
                                      X[y == 'rest'], axis=0)
        # Use a log scale for p-values
        log_p_values = -np.log10(p_values)
        
        # NAN values to zero
        log_p_values[np.isnan(log_p_values)] = 0.0
        log_p_values[log_p_values > 10.0] = 10.0
        
        # threshold the p values and binarize
        log_p_values = np.where(log_p_values > 2.0, log_p_values, 0.0)
        mask = np.where(log_p_values > 0.0, 1.0, 0.0)
        

        # save mask 
        mask_img = new_img_like(data_loader.reference_img, mask)
        mask_img.to_filename(f'data/HCP_LANGUAGE/subj{subj}/mask_p0.01.nii.gz')
        
        # dilate the mask
        mask = binary_dilation(mask)
        mask_img_dilated = new_img_like(data_loader.reference_img, mask)
        mask_img_dilated.to_filename(f'data/HCP_LANGUAGE/subj{subj}/mask_p0.01_dilated.nii.gz')
from nilearn.image import load_img, binarize_img, index_img, clean_img, new_img_like
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict


class GTZANDataLoader():
    # Load data from local preprocessed nii.gz files
    def __init__(self, data_dir='data/ds003720-derivative', subject=[1], mask_file=None, threshold=0.9, runs=None):
        self.data_dir = data_dir
        self.subject = [int(sub) for sub in subject] if isinstance(subject, list) else [int(subject)]
        self.runs = runs if runs is not None else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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
        nii_files = glob.glob(os.path.join(self.data_dir, f'sub-{subject[0]:03d}/func', f'*MNI152NLin2009cAsym_desc-preproc_bold.nii'))
        
        self.reference_img = index_img(load_img(nii_files[0]), 0)
        
        
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
            
            # get nii files which contain "MNI152NLin2009cAsym_desc-preproc_bold.nii"
            # the filename will look like: "sub-002_task-Test_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
            nii_files = glob.glob(os.path.join(self.data_dir, f'sub-{subject:03d}/func', f'*MNI152NLin2009cAsym_desc-preproc_bold.nii'))
            # get the run information from the nii file names eg. Test_run-01, Test_run-02, Train_run-01, Train_run-02 etc.
            runs = [nii_file.split('task-', 1)[1].split('_space', 1)[0] for nii_file in nii_files]
            # filter out the runs that are not in self.runs
            runs = [run for run in runs if int(run.split('_run-', 1)[1]) in self.runs]

            
            print('nii_files:', nii_files)
            print('runs:', runs)            
            
        
            
            labels_runs = {}
            # data/ds003720-derivative/sub-001/func/sub-001_task-Test_run-01_events.tsv
            for run in runs:
                labels = pd.read_csv(os.path.join(self.data_dir, f'sub-{subject:03d}/func', f'sub-{subject:03d}_task-{run}_events.tsv'), sep='\t')
                
                # columns are 'onset', 'duration','trial_type'
                # divide onset and duration by 1.5 (TR=1.5), so that we can get the label of each volume
                # eg. for volumes 1-30, the label is happy, for volumes 31-60, the label is sad, etc.
                labels['onset'] = labels['onset'].apply(lambda x: float(x) / 1.5)
                labels['duration'] = labels['duration'].apply(lambda x: float(x) / 1.5)
                # get the label of each volume for this run
                labels_runs[run] = np.repeat(labels['genre'].values, labels['duration'].values.astype(int))

            fourD_data_runs = {}
            for run in runs:
                # load 4D data
                fourD_data_runs[run] = load_img(os.path.join(self.data_dir, f'sub-{subject:03d}/func', f'sub-{subject:03d}_task-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii'))
            
                # Masking
                if self.mask is not None:
                    masker = NiftiMasker(mask_img=self.mask, standardize=True)
                    fourD_data_runs[run] = masker.fit_transform(fourD_data_runs[run])
                else:
                    # Just standardize the data if mask is not specified
                    fourD_data_runs[run] = clean_img(fourD_data_runs[run]).get_fdata().transpose(3, 0, 1, 2)
                    # fourD_data_runs[run] = fourD_data_runs[run].get_fdata().transpose(3, 0, 1, 2)
                    pass
                print(f'fourD_data_runs[{run}].shape:', fourD_data_runs[run].shape)
            
            
            
            
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
    
    # calculate tsnr for each subject and save it to a csv file
    import pandas as pd
    df = pd.DataFrame(columns=['subject', 'tsnr'])
    for subj in range(1, 6):
        data_loader = GTZANDataLoader(subject=[subj])
        X, y = data_loader.load_volume_data()
        print('label counts:', np.unique(y, return_counts=True))
        tsnr = get_tsnr(X)
        df = pd.concat([df, pd.DataFrame({'subject': [subj], 'tsnr': [tsnr]})], ignore_index=True)
        
    df.to_csv('data/ds003720-derivative/tsnr.csv', index=False)
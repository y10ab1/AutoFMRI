from nilearn.image import load_img, binarize_img, index_img, clean_img, new_img_like
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
import os
import numpy as np
import pandas as pd
from collections import defaultdict

class EmoDataLoader():
    # Load data from local preprocessed nii.gz files
    def __init__(self, data_dir='data/fmri-dataset-for-emotion-recognition', subject=[1], mask_file=None, threshold=0.9, runs=None):
        self.data_dir = data_dir
        self.subject = [int(sub) for sub in subject] if isinstance(subject, list) else [int(subject)]
        self.runs = runs if runs is not None else [1, 2, 3, 4, 5]
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
            
        self.reference_img = index_img(os.path.join(self.data_dir, f'Sub-{self.subject[0]:02d}', f'wrsub-{self.subject[0]:02d}_task-emotionalfaces_run-1_bold.nii'), 0)
        
        
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
            
            labels_runs = {}
            for i in self.runs:
                labels = pd.read_csv(os.path.join(self.data_dir, 'onsetime', f'task-emotionalfaces_run-{i}_events.tsv'), sep='\t')
                
                # get head 13 rows, columns are 'onset', 'duration','trial_type'
                labels = labels.head(13)
                # divide onset and duration by 2 (TR=2), so that we can get the label of each volume
                # eg. for volumes 1-30, the label is happy, for volumes 31-60, the label is sad, etc.
                labels['onset'] = labels['onset'].apply(lambda x: float(x) / 2)
                labels['duration'] = labels['duration'].apply(lambda x: float(x) / 2)
                # get the label of each volume for this run
                labels_runs[i] = [labels['trial_type'][j] for j in range(13) for _ in range(int(labels['duration'][j]))]

            fourD_data_runs = {}
            for i in self.runs:
                fourD_data_runs[i] = load_img(os.path.join(self.data_dir, f'Sub-{subject:02d}', f'wrsub-{subject:02d}_task-emotionalfaces_run-{i}_bold.nii'))

            
                # Masking
                if self.mask is not None:
                    masker = NiftiMasker(mask_img=self.mask, standardize=True)
                    fourD_data_runs[i] = masker.fit_transform(fourD_data_runs[i])
                else:
                    # Just standardize the data if mask is not specified
                    fourD_data_runs[i] = clean_img(fourD_data_runs[i]).get_fdata().transpose(3, 0, 1, 2)
                    # fourD_data_runs[i] = fourD_data_runs[i].get_fdata().transpose(3, 0, 1, 2)
                    pass
                print(f'fourD_data_runs[{i}].shape:', fourD_data_runs[i].shape)
            
            
            
            
            # volume_data = fourD_data.get_fdata()
            
            # (n_samples, n_x, n_y, n_z) if mask is None, otherwise (n_samples, n_voxels)
            X_sub = np.concatenate([fourD_data_runs[i] for i in self.runs], axis=0)
            y_sub = np.concatenate([labels_runs[i] for i in self.runs], axis=0)
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

    # fMRIDataLoader = EmoDataLoader(data_dir='data/fmri-dataset-for-emotion-recognition', subject=[1],
    #                                   mask_file='data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii', threshold=0.9)
    # X, y = fMRIDataLoader.load_volume_data()
    # exit()


    # # calculate tsnr
    # tsnr = get_tsnr(X)
    
    
    # # save epi mask
    # whole_brain_mask = compute_epi_mask(fMRIDataLoader.reference_img)
    # whole_brain_mask.to_filename('data/whole_brain_mask.nii')
    
    # calculate tsnr for each subject and save it to a csv file
    import pandas as pd
    df = pd.DataFrame(columns=['subject', 'tsnr'])
    for subj in range(1, 6):
        data_loader = EmoDataLoader(data_dir='data/fmri-dataset-for-emotion-recognition', subject=[subj])
                                      
        X, y = data_loader.load_volume_data()
        tsnr = get_tsnr(X)
        df = df.append({'subject': subj, 'tsnr': tsnr}, ignore_index=True)
        
    df.to_csv('data/fmri-dataset-for-emotion-recognition/tsnr.csv', index=False)
from nilearn.image import load_img, binarize_img, index_img, clean_img, resample_img, new_img_like
from nilearn.maskers import NiftiMasker
import os
import numpy as np
import pandas as pd
import scipy

class VKMCDataLoader():
    # Load data from local preprocessed nii.gz files
    def __init__(self, data_dir, subject, mask_file=None, threshold=None, label_type='midi', augment=0):
        self.data_dir = data_dir
        self.subject = [subject] if isinstance(subject, str) else subject # subject can be a string or a list of strings
        self.label_type = label_type  # 'midi' or 'timbre'
        self.mask = load_img(mask_file) if mask_file is not None else None
        self.augment = augment
        print('Mask shape:', self.mask.shape if self.mask is not None else None)
        
        # if mask is not made of 0s and 1s, threshold it
        if threshold is not None:
            self.mask = binarize_img(self.mask, threshold)
            
            print('Mask shape after thresholding:', self.mask.shape, 
                  '0s:', np.sum(self.mask.get_fdata() == 0),
                    '1s:', np.sum(self.mask.get_fdata() == 1))
            
        # self.reference_img = load_img('data/2021_Reconstruction_dataset/masks/ACmask.nii')
        self.reference_img = None
        


    def load_volume_data(self):
        # Load 4D data from data_dir and get labels
        X, y = [], []
        for subject in self.subject:
            
            # Load and parse the stimuli 2 sessions log files
            
            onset_times_and_labels = {
                'session-1' : pd.read_csv(os.path.join(self.data_dir, 'logfiles-AB', f'{subject}_ses-1.csv')),
                'session-2' : pd.read_csv(os.path.join(self.data_dir, 'logfiles-AB', f'{subject}_ses-2.csv')),
            }
            
            
            # load all nii data for one subject
            for niifile in os.listdir(os.path.join(self.data_dir, subject, 'func')):
                run = niifile.split('run-')[1].split('.')[0]
                session = niifile.split('ses-')[1].split('_')[0]
                
                if niifile.endswith('.nii') and session in ['1', '2'] and run in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    # eg. sub-001_ses-1_run-1.nii the number before .nii is the run number which should be in [1, 8]
                    fourD_data = load_img(os.path.join(self.data_dir, subject, 'func', niifile))
                    
                    print('Loading:', niifile)
                else:
                    continue
                
                if self.reference_img is None:
                    self.reference_img = index_img(fourD_data, 0)

                # Masking
                if self.mask is not None:
                    masker = NiftiMasker(mask_img=self.mask, 
                                         standardize='zscore_sample',
                                         )
                    fourD_data = masker.fit_transform(fourD_data)
                else:
                    # Just standardize the data if mask is not specified
                    fourD_data = clean_img(fourD_data)
                    # resample the data to the same shape as the reference image (ACmask.nii for now)
                    # fourD_data = resample_img(fourD_data, 
                    #                             target_affine=self.reference_img.affine,
                    #                             target_shape=self.reference_img.shape,
                    #                             interpolation='nearest')

                print('fourD_data.shape:', fourD_data.shape)
        
                

                # Assign labels to volumes
                # Note: We assume that the TR (repetition time) is 2 seconds, 
                # TA (acquisition time) is 1 second, and the time between each stimulus is around 6 seconds.
                labels = []
                if isinstance(fourD_data, np.ndarray): # masked data (n_samples, n_features)
                    volume_times = np.arange(0, fourD_data.shape[0]*2, 2)
                else: # unmasked data (n_samples, n_x, n_y, n_z)
                    volume_times = np.arange(0, fourD_data.shape[-1]*2, 2)  # The times at which each volume was acquired
                for t in volume_times:
                    # Find the stimulus that started before this volume was acquired, and with the right run
                    onset_times_and_labels_df = onset_times_and_labels[f'session-{session}']
                    onset_times_and_labels_df = onset_times_and_labels_df[onset_times_and_labels_df['run'] == int(run)]
                                        
                    stimuli = onset_times_and_labels_df.loc[  # Get the stimuli that started before this volume
                        (onset_times_and_labels_df['sound.started'] <= t) & # Make sure the stimuli started before this volume
                        (onset_times_and_labels_df['sound.started'] + 6 >= t) # Make sure the stimuli started within 6 seconds before this volume
                    ]
                    
                    # If there is at least one stimulus, get the last one. Otherwise, assign the 'rest' label to this volume.
                    if not stimuli.empty:
                        stimulus = stimuli.iloc[-1]
                        labels.append(stimulus['sound.' + self.label_type])
                    else:
                        labels.append('rest')
                    
                # Add the data and labels of this run to the lists

                if isinstance(fourD_data, np.ndarray): # masked data (n_samples, n_features)
                    # X = np.concatenate((X, fourD_data), axis=0) 
                    X.append(fourD_data)
                else: # unmasked data (n_samples, n_x, n_y, n_z)
                    # X = np.concatenate((X, np.array([index_img(fourD_data, i).get_fdata() for i in range(fourD_data.shape[-1])])), axis=0)
                    X.append(np.array([index_img(fourD_data, i).get_fdata() for i in range(fourD_data.shape[-1])]))
                # y = np.concatenate((y, labels), axis=0)
                y.append(labels)

        
        
        
        # Convert X and y to numpy arrays, 
        
        X = np.array(X, dtype=np.float32).squeeze() # (n_runs, n_samples, n_features) or (n_runs, n_samples, n_x, n_y, n_z)
        y = np.array(y) # (n_runs, n_samples)
              
        # remove run dimension 
        # X will be (n_samples, n_features) or (n_samples, n_x, n_y, n_z)
        # y will be (n_samples,)
        
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        
        
        
        # Using Scipy to interpolate the between volumes to get 1 volume per second
        if self.augment > 0:
            zoom_factor = (self.augment, 1) if X.ndim == 2 else (self.augment, 1, 1, 1)
            X = scipy.ndimage.zoom(X, zoom_factor, order=1)
            y = np.repeat(y, self.augment)
            
        # convert nan to 0 and make sure there is no nan in X
        X = np.nan_to_num(X)
        assert np.sum(np.isnan(X)) == 0
            
        

        print('X.shape:', X.shape, 'y.shape:', y.shape)
        assert X.shape[0] == y.shape[0] # Make sure the number of samples is the same for X and y

        return X, y
    
    def load_data(self):
        # Load beta maps from data_dir and get labels
        # default data_dir: AutoFMRI/data/datas/LSA_all
        # each beta maps: AutoFMRI/data/datas/LSA_all/sub-001/bass_electronic_018-062-12.nii
        self.data_dir = 'data/datas/LSA_all'
        
        if self.mask is not None:
            masker = NiftiMasker(mask_img=self.mask, 
                                    standardize='zscore_sample',
                                    )
        else:
            masker = NiftiMasker(standardize='zscore_sample',
                                    )
        
        X, y = [], []
        for subject in self.subject:
            for nii_file in os.listdir(os.path.join(self.data_dir, subject)):
                if nii_file.endswith('.nii'):
                    # eg. bass_electronic_018-062-12.nii
                    # intrument: bass, timbre: electronic, midi: 62, run: 12

                    
                    label_dict = {'midi': nii_file.split('_')[2].split('-')[1],
                                  'timbre': nii_file.split('_')[1],
                                  'instrument': nii_file.split('_')[0],
                    }    
                    
                    label = label_dict[self.label_type]
                    
                    # print(label)
                    # print(os.path.join(self.data_dir, subject, nii_file))
                    
                    # masking if mask is not None
                    if self.mask is not None:
                        fourD_data = masker.fit_transform(load_img(os.path.join(self.data_dir, subject, nii_file)))
                        
                    if self.reference_img is None:
                        self.reference_img = load_img(os.path.join(self.data_dir, subject, nii_file)) # load the first image as reference image
                    
                    X.append(load_img(os.path.join(self.data_dir, subject, nii_file)).get_fdata())
                    
                    y.append(label)
        
                    
        
        # Convert X and y to numpy arrays,
        
        # convert nan to 0
        
        
        X = np.array(X, dtype=np.float32).squeeze() # (n_samples, n_features) or (n_samples, n_x, n_y, n_z)
        X = np.nan_to_num(X)
        y = np.array(y) # (n_samples)
        
        assert X.shape[0] == y.shape[0] # Make sure the number of samples is the same for X and y
            
        return X, y
        
        
    
if __name__ == '__main__':
    # Test the modified VKMCDataLoader with one subject and one run
    data_loader = VKMCDataLoader(
                                data_dir='data/2021_Reconstruction_dataset',
                                # data_dir='data/datas/LSA_all',
                                # subject=['sub-001', 'sub-002'],
                                subject=['sub-001'],
                                mask_file='data/2021_Reconstruction_dataset/masks/ACmask.nii',
                                augment=True,
    )
    
    X, y = data_loader.load_volume_data()
    # X, y = data_loader.load_data()
    
    # print first 30 labels
    print(y[:30])
    
    # count the number of labels
    # print('Number of labels:', len(np.unique(y)))
    print('Counts of each label:', np.unique(y, return_counts=True))
    
    print('X.shape:', X.shape, 'y.shape:', y.shape)
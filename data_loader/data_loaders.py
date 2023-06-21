from nilearn.image import load_img, binarize_img
from nilearn.maskers import NiftiMasker
import os
import numpy as np


class HaxbyDataLoader():
    # Load data from local preprocessed nii.gz files
    def __init__(self, data_dir, subject, mask_file=None, threshold=None):
        self.data_dir = data_dir
        self.subject = subject
        self.mask = load_img(mask_file) if mask_file is not None else None
        print('Mask shape:', self.mask.shape if self.mask is not None else None)
        
        # if mask is not made of 0s and 1s, threshold it
        if threshold is not None:
            self.mask = binarize_img(self.mask, threshold)
            print('Mask shape after thresholding:', self.mask.shape, 
                  '0s:', np.sum(self.mask.get_fdata() == 0),
                    '1s:', np.sum(self.mask.get_fdata() == 1))
            
            
            
        
        # Specify a image from data_dir as the reference image
        self.reference_img = load_img(os.path.join(self.data_dir, sorted(os.listdir(self.data_dir))[0]))
        
        self.X, self.y = self.load_data()
        
        
    def load_data(self):
        # Each image file name is like: bottle_10.nii.gz, where bottle is the label and 10 is the run number
        
        # Load data from data_dir
        # X: 4D array of shape (n_samples, n_x, n_y, n_z)
        # y: 1D array of shape (n_samples,)
        X, y = [], []
        for img_file in sorted(os.listdir(self.data_dir)):
            if img_file.endswith('.nii.gz'):
                img = load_img(os.path.join(self.data_dir, img_file))
                
                # Masking
                if self.mask is not None:
                    masker = NiftiMasker(mask_img=self.mask, standardize=True)
                    img = masker.fit_transform(img)
                
                X.append(img.get_fdata() if not isinstance(img, np.ndarray) else img)
                y.append(img_file.split('_')[0])
                
        # Convert X and y to numpy arrays
        X = np.array(X, dtype=np.float32).squeeze()
        y = np.array(y)
        

        print('X.shape:', X.shape, 'y.shape:', y.shape)
        
        return X, y
    
    
    
if __name__ == '__main__':
    X, y = HaxbyDataLoader(data_dir='data/haxby2001/subj4/first_level_output', subject=4).load_data()
    y = y.astype(np.int)
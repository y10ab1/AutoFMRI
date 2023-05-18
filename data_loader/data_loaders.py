from nilearn.image import load_img
import os
import numpy as np


class HaxbyDataLoader():
    # Load data from local preprocessed nii.gz files
    def __init__(self, data_dir, subject, label_encoder=None):
        self.data_dir = data_dir
        self.subject = subject
        self.label_encoder = label_encoder

        
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
                X.append(img.get_fdata())
                y.append(img_file.split('_')[0])
                
        # Convert X and y to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        if self.label_encoder is not None:
            y = self.label_encoder.fit_transform(y)
        
        print('X.shape:', X.shape, 'y.shape:', y.shape)
        
        return X, y
    
    
if __name__ == '__main__':
    X, y = HaxbyDataLoader(data_dir='data/haxby2001/subj4/first_level_output', subject=4).load_data()
    y = y.astype(np.int)
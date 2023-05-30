# load vt mask from nilearn and check its size

import os
import numpy as np
from nilearn.image import load_img



# load vt mask
vtmask = load_img('data/haxby2001/subj4/mask4_vt.nii')
vtmask_data = vtmask.get_fdata()
print(vtmask_data.shape)

# check valid voxels
valid_voxels = np.where(vtmask_data != 0)
print(len(valid_voxels), valid_voxels)
print(valid_voxels[1].shape)
from nilearn.image import get_data, new_img_like
import nibabel as nib
import os
import numpy as np


def union_images(images):
    # images is a list of nilearn images
    # return a new image that is the union of all images
    # the new image has the same shape as the first image in the list
    # the new image has 1 at a voxel if at least one image in the list has 1 at that voxel
    # the new image has 0 at a voxel if all images in the list have 0 at that voxel
    union = get_data(images[0])
    for i in range(1, len(images)):
        union = np.logical_or(union, get_data(images[i]))
    return new_img_like(images[0], union)


if __name__ == '__main__':
    # load fMRI images from a directory
    # fmri_dir = 'result-10x10x10-40patches-subj4'
    fmri_dir = 'icassp-results-0602/icassp_haxby2001_subj5_rf_rf-1'
    file_name_prefix = 'high_performance_voxels_mask'
    file_name_prefix = 'selected_patch_masks'
    fmri_images = []
    for filename in os.listdir(fmri_dir):
        if filename.endswith('.nii') and filename.startswith(file_name_prefix):
            fmri_images.append(nib.load(os.path.join(fmri_dir, filename)))
            
    # union all images
    union_image = union_images(fmri_images)
    
    # save the union image
    nib.save(union_image, os.path.join(fmri_dir, f'{file_name_prefix}-union.nii'))
    
    
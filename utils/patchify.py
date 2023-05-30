import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import get_data, new_img_like
import nibabel as nib

def patchify_by_cube(X, cube_size=(20, 20, 20), stride=(20, 20, 20)):
    # X is a 4D array of shape (n_samples, n_x, n_y, n_z)
    # return a list of patch masks, each patch mask is a list of indices where the mask is 1
    
    reference_img = X[0]
    patch_masks = []
    # patchify the reference image by cube
    for x in range(0, reference_img.shape[0], stride[0]):
        for y in range(0, reference_img.shape[1], stride[1]):
            for z in range(0, reference_img.shape[2], stride[2]):
                # if the non-zero elements in the original image is less than 
                # 10% of the total number of elements in the patch, then skip this patch
                if np.count_nonzero(reference_img[x:x+cube_size[0], y:y+cube_size[1], z:z+cube_size[2]]) < 0.1 * np.prod(cube_size):
                    continue
                
                patch_mask = np.zeros(reference_img.shape, dtype=np.int)
                patch_mask[x:x+cube_size[0], y:y+cube_size[1], z:z+cube_size[2]] = 1
                patch_masks.append(np.where(patch_mask == 1))
    
    print('Number of patches:', len(patch_masks), 'Patch size:', patch_masks[0][0].shape)
    return patch_masks


def patchify_by_atlas(X, atlas_name='juelich'):
    # X is a 4D array of shape (n_samples, n_x, n_y, n_z)
    # return a list of patch masks, each patch mask is a list of indices where the mask is 1

    # Fetch the atlas
    if atlas_name == 'harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif atlas_name == 'juelich':
        atlas = datasets.fetch_atlas_juelich('prob-2mm')
    else:
        raise ValueError('Invalid atlas name')

    # Get atlas data
    atlas_data = get_data(atlas.maps)
    print('Atlas data shape:', atlas_data.shape)
    print('Number of labels:', len(np.unique(atlas_data)), 'Labels:', np.unique(atlas_data))

    patch_masks = []
    for label in np.unique(atlas_data)[1:]:  # start from 1 to exclude background
        # Create a mask for the current label
        mask = (atlas_data == label)
        print(mask.shape)
        print('Label:', label, 'Number of voxels:', np.sum(mask))

        # if the non-zero elements in the original image is less than 
        # 10% of the total number of elements in the patch, then skip this patch
        if np.count_nonzero(X[0][mask]) < 0.1 * np.sum(mask):
            continue

        # Get the indices where the mask is 1
        indices = np.where(mask)
        patch_masks.append(indices)

    print('Number of patches:', len(patch_masks), 'Patch size:', patch_masks[0][0].shape)
    return patch_masks

def get_top_k_patches(patch_scores, patch_masks, X_train, topk_patches, ref_niimg=None, output_filename=None):
    print('Top k performance scores:', np.sort(patch_scores)[-topk_patches:])
    
    selected_patch_masks_idx = np.argsort(patch_scores)[-topk_patches:]
    selected_patch_masks = np.array(patch_masks, dtype=object)[selected_patch_masks_idx].tolist()
    
    high_performance_voxels_mask = np.zeros(X_train[0].shape)
    
    # Save selected patches as a 3D image
    # The shape of selected_patch_masks is (n_samples, x_indices, y_indices, z_indices)
    if output_filename:
        for i, patch_mask in enumerate(selected_patch_masks):
            # assign value according to the patch score
            high_performance_voxels_mask[patch_mask[0], patch_mask[1], patch_mask[2]] = patch_scores[selected_patch_masks_idx[i]]
        # nib.save(nib.Nifti1Image(high_performance_voxels_mask, X_train[1].affine), output_filename)
        nifti_img = new_img_like(ref_niimg=ref_niimg,
                                data=high_performance_voxels_mask).to_filename(output_filename)
    high_performance_voxels_mask = np.zeros(X_train[0].shape)
        
    return selected_patch_masks, high_performance_voxels_mask


    
    
if __name__ == '__main__':
    import nibabel as nib
    import matplotlib.pyplot as plt

    # Load the image
    img = nib.load('data/haxby2001/subj2/preprocessed/swubold.nii')

    # Get the data
    X = get_data(img)
    
    # reshape the data to (n_samples, n_x, n_y, n_z)
    X = np.transpose(X, (3, 0, 1, 2))

    # Patchify the image
    patch_masks = patchify_by_atlas(X, atlas_name='juelich')

    # Visualize the first patch
    plt.imshow(X[0][patch_masks[0]])
    plt.savefig('patch_example.png')
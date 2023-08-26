import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional


from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import get_data, new_img_like, resample_img, load_img



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


def patchify_by_atlas(X, reference_img, atlas_name='yeo', threshold=0.1, verbose=False):
    # X is a 4D array of shape (n_samples, n_x, n_y, n_z)
    # reference_img is a Nifti1Image object
    # return a list of patch masks, each patch mask is a list of indices where the mask is 1
    
    print('Reference image shape (3D):', reference_img.shape)

    # Fetch the atlas
    if atlas_name == 'harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        
        # Transform the 3D atlas into a 4D atlas, each 3D volume represents a region
        atlas_maps = []
        for label in np.unique(get_data(atlas.maps))[1:]:
            atlas_maps.append(get_data(atlas.maps) == label)
        atlas_data = np.stack(atlas_maps, axis=-1)
    elif atlas_name == 'juelich':
        atlas = datasets.fetch_atlas_juelich('prob-2mm')
        atlas_data = get_data(atlas.maps)
    elif atlas_name == 'yeo':
        atlas = datasets.fetch_atlas_yeo_2011()
        atlas['maps'] = atlas['thick_17']
        atlas['labels'] = list(np.unique(get_data(atlas['maps'])))
        
        atlas_maps = []
        for label in np.unique(get_data(atlas.maps))[1:]:
            atlas_maps.append(get_data(atlas.maps)[..., 0] == label)
        atlas_data = np.stack(atlas_maps, axis=-1)
    elif atlas_name == 'yeo400':
        # load atlas file from local atlas folder
        atlas = {}
        atlas['maps'] = nib.load('atlas/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii')
        atlas['labels'] = list(np.unique(get_data(atlas['maps'])))
        atlas = type('atlas', (object,), atlas)() # convert dict to object
        
        # Transform the 3D atlas into a 4D atlas, each 3D volume represents a region
        atlas_maps = []
        for label in np.unique(get_data(atlas.maps))[1:]:
            atlas_maps.append(get_data(atlas.maps) == label)
        atlas_data = np.stack(atlas_maps, axis=-1)

    elif atlas_name == 'hcp_mmp_sym':
        # load atlas file from local atlas folder
        atlas = {}
        atlas['maps'] = nib.load('atlas/MMP_in_MNI_symmetrical_1.nii')
        atlas['labels'] = list(np.unique(get_data(atlas['maps'])))
        atlas = type('atlas', (object,), atlas)() # convert dict to object
        
        # Transform the 3D atlas into a 4D atlas, each 3D volume represents a region
        atlas_maps = []
        for label in np.unique(get_data(atlas.maps))[1:]:
            atlas_maps.append(get_data(atlas.maps) == label)
        atlas_data = np.stack(atlas_maps, axis=-1)
        
    else:
        raise ValueError('Invalid atlas name')
    

    # Resample the atlas to the same affine as the reference image
    atlas_resampled = resample_img(new_img_like(atlas.maps, atlas_data), 
                                    target_affine=reference_img.affine, 
                                    target_shape=reference_img.shape,
                                    interpolation='nearest')
    atlas_data = get_data(atlas_resampled)

    print('Regions labels:', atlas.labels, 'Number of regions:', len(atlas.labels))
    print('Atlas data shape:', atlas_data.shape)

    patch_masks = []
    for i in range(atlas_data.shape[-1]):  # loop over regions in the 4D atlas
        # Create a mask for the current region
        mask = atlas_data[..., i].astype(bool)
        
        if verbose:
            print('Region:', atlas.labels[i+1], 'Number of voxels:', np.sum(mask))

        # if a patch contains less than threshold (default: 10%) of the non-zero voxels in the region, then skip this patch
        if np.count_nonzero(reference_img.get_fdata()[mask]) <= threshold * np.sum(mask):
            continue

        # Get the indices where the mask is 1
        indices = np.where(mask)
        if verbose:
            print('Indices:', indices)
        
        # Also skip the patch if the number of voxels is empty
        if len(indices[0]) == 0:
            continue
        patch_masks.append(indices)

    print('Number of candidate patches:', len(patch_masks))
    return patch_masks

# def get_top_k_patches(patch_scores, patch_masks, X_train, topk_patches, ref_niimg=None, output_filename=None):
#     topk_patches = int(topk_patches)
#     print('Top k performance scores:', np.sort(patch_scores)[-topk_patches:])
    
#     selected_patch_masks_idx = np.argsort(patch_scores)[-topk_patches:]
#     selected_patch_masks = np.array(patch_masks, dtype=object)[selected_patch_masks_idx].tolist()
    
#     high_performance_voxels_mask = np.zeros(X_train[0].shape)
    
#     # Save selected patches as a 3D image
#     # The shape of selected_patch_masks is (n_samples, x_indices, y_indices, z_indices)
#     if output_filename:
#         for i, patch_mask in enumerate(selected_patch_masks):
#             # assign value according to the patch score
#             high_performance_voxels_mask[patch_mask[0], patch_mask[1], patch_mask[2]] = patch_scores[selected_patch_masks_idx[i]]
#         # nib.save(nib.Nifti1Image(high_performance_voxels_mask, X_train[1].affine), output_filename)
#         nifti_img = new_img_like(ref_niimg=ref_niimg,
#                                 data=high_performance_voxels_mask).to_filename(output_filename)
#     high_performance_voxels_mask = np.zeros(X_train[0].shape)
        
#     return selected_patch_masks, high_performance_voxels_mask



def get_top_k_patches(patch_scores: np.array, patch_masks: np.array, X_train: np.array, topk_patches: int, ref_niimg=None, output_filename: Optional[str]=None) -> Tuple[List, np.array]:
    topk_patches = int(topk_patches)
    print('Top k performance scores:', np.sort(patch_scores)[-topk_patches:])
    
    selected_patch_masks_idx = np.argsort(patch_scores)[-topk_patches:]
    selected_patch_masks = np.array(patch_masks, dtype=object)[selected_patch_masks_idx].tolist()
    
    high_performance_voxels_mask = np.zeros(X_train[0].shape)

    if output_filename:
        save_selected_patches_as_3D_image(selected_patch_masks, patch_scores, selected_patch_masks_idx, ref_niimg, high_performance_voxels_mask, output_filename)
    high_performance_voxels_mask = np.zeros(X_train[0].shape)
        
    return selected_patch_masks, high_performance_voxels_mask


def save_selected_patches_as_3D_image(selected_patch_masks: List, patch_scores: np.array, selected_patch_masks_idx: np.array, ref_niimg, high_performance_voxels_mask: np.array, output_filename: str) -> None:
    # Save selected patches as a 3D image
    # The shape of selected_patch_masks is (n_samples, x_indices, y_indices, z_indices)
    for i, patch_mask in enumerate(selected_patch_masks):
        # assign value according to the patch score
        high_performance_voxels_mask[patch_mask[0], patch_mask[1], patch_mask[2]] = patch_scores[selected_patch_masks_idx[i]]
    nifti_img = new_img_like(ref_niimg=ref_niimg,
                            data=high_performance_voxels_mask).to_filename(output_filename)



def integer_to_binary_list(num, length=10):
    binary_string = bin(num)[2:]  # Convert integer to binary string and remove the '0b' prefix
    binary_list = [0] * (length - len(binary_string)) + [int(bit) for bit in binary_string]
    return binary_list


def apply_mask(mask, data):
    masked_data = [x for x, m in zip(data, mask) if m == 1]
    return masked_data



    
    
if __name__ == '__main__':

    for idx in range(1, 6):
        # Load the image
        path = f'data/haxby2001/wmasks_binary/wmask4_vt_subj{idx}.nii'
        img = nib.load(path)
        print('Image shape:', img.shape)
            
        # patchify the image by atlas
        patch_masks = patchify_by_atlas(img.get_fdata(), img, atlas_name='yeo', threshold=0.05)
        
        print('Number of patches:', len(patch_masks))
        for i in range(len(patch_masks)):
            print('Patch', i, 'shape:', patch_masks[i][0].shape)
            
            
        # create new 3D binary image
        new_img = np.zeros(img.shape)
        for i in range(len(patch_masks)):
            new_img[patch_masks[i]] = 1
            
        # save the new image
        nib.save(nib.Nifti1Image(new_img, img.affine), f'data/haxby2001/wmasks_binary/wmask4_vt_subj{idx}_atlas_yeo.nii')
        
    
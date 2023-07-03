import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import get_data, new_img_like, resample_img



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


def patchify_by_atlas(X, reference_img, atlas_name='juelich'):
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
        print('Region:', atlas.labels[i+1], 'Number of voxels:', np.sum(mask))

        # if the non-zero elements in the original image is less than 
        # 10% of the total number of elements in the patch, then skip this patch
        if np.count_nonzero(reference_img.get_fdata()[mask]) < 0.1 * np.sum(mask):
            continue

        # Get the indices where the mask is 1
        indices = np.where(mask)
        patch_masks.append(indices)

    print('Number of patches:', len(patch_masks))
    return patch_masks

def get_top_k_patches(patch_scores, patch_masks, X_train, topk_patches, ref_niimg=None, output_filename=None):
    topk_patches = int(topk_patches)
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

    # Load the image
    img = nib.load('data/haxby2001/subj2/preprocessed/swubold.nii')
    reference_img = img.slicer[..., 0]
    # Get the data
    X = get_data(img)
    
    # reshape the data to (n_samples, n_x, n_y, n_z)
    X = np.transpose(X, (3, 0, 1, 2))

    # Patchify the image
    patch_masks = patchify_by_atlas(X, reference_img, atlas_name='hcp_mmp_sym')
    # patch_masks = patchify_by_atlas(X, reference_img, atlas_name='harvard_oxford')

    # Save some patch masks as 3D images and also aggregate them  and save into a 3D image
    aggregate_patch_mask = np.zeros(X[0].shape)
    for i, patch_mask in enumerate(patch_masks[:40:3]):
        patch_masks_img = np.zeros(X[0].shape)
        patch_masks_img[patch_mask[0], patch_mask[1], patch_mask[2]] = 1
        nib.save(nib.Nifti1Image(patch_masks_img, img.affine), f'patch_mask_{i}.nii')
        
        aggregate_patch_mask[patch_mask[0], patch_mask[1], patch_mask[2]] = 1
        
    nib.save(nib.Nifti1Image(aggregate_patch_mask, img.affine), 'aggregate_patch_mask.nii')
        
    
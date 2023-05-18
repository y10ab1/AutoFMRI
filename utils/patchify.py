import numpy as np


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
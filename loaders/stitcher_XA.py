import tifffile
import importlib
import sys, os
import numpy as np
import module_auxiliary as ma



# Info: Computed start indices are for FBP: [350, 94, 94, 94, 94, 94, 94]
# Computed end indices are [693, 694, 694, 694, 694, 693, 788]
# Folders are [1,2,...,7]

# For registerring to FOV of Neutron recon, use for FBP
# Computed start indices are: [350, 94, 94, 94, 94]
# Computed end indices are [693, 694, 694, 694, 788]
# Folders are [1,2,3,4,5]

def stich_indices(dataset = 'fbp', folders = [1,2,3,4,5,6,7]):
    d = 50
    D = 250
    generic_f_tv = '/dtu-compute/msaca/sliceA_xray_pc/output/tv_recon/r_###.'
    generic_f_tvs = ma.generate_paths(generic_f_tv,folders)

    generic_f_fbp ='/dtu-compute/msaca/sliceA_xray_pc/output/fbp_recon/r_###.'
    generic_f_fbps = ma.generate_paths(generic_f_fbp,folders)

    start_indices = [0]
    end_indices = []
    for folder_idx in range(len(folders)-1):
        if dataset == 'fbp':
            generic_path_pre = generic_f_fbps[folder_idx][:-1]+'/slice_fbp_####.tiff'
        elif dataset == 'tv':
            generic_path_pre = generic_f_tvs[folder_idx][:-1]+'/slice_tv_####.tiff'

        if dataset == 'fbp':
            generic_path_post = generic_f_fbps[folder_idx+1][:-1]+'/slice_fbp_####.tiff'
        elif dataset == 'tv':
            generic_path_post = generic_f_tvs[folder_idx+1][:-1]+'/slice_tv_####.tiff'
        
        N_slices = ma.find_largest_number2(generic_path_pre)
        path_pre = ma.generate_paths(generic_path_pre,[N_slices - d])

        A = np.arange(D)
        path_post = ma.generate_paths(generic_path_post,A)
        image_pre = tifffile.imread(path_pre[0])
        norms = np.empty(D)
        for i in A:
            image = tifffile.imread(path_post[i])
            norms[i] = np.linalg.norm(image_pre[150:650] - image[150:650],ord=1)
        w = np.argmin(norms)
        end_indices.append(N_slices - int(np.floor((w+d)/2))-1)
        start_indices.append(int(np.ceil((w+d)/2)))
    end_indices.append(N_slices)
    return start_indices, end_indices



def generate_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5,6,7], start_indices = None, end_indices = None):
    generic_f_tv = '/dtu-compute/msaca/sliceA_xray_pc/output/tv_recon/r_###.'
    generic_f_tvs = ma.generate_paths(generic_f_tv,folders)

    generic_f_fbp ='/dtu-compute/msaca/sliceA_xray_pc/output/fbp_recon/r_###.'
    generic_f_fbps = ma.generate_paths(generic_f_fbp,folders)

    paths = []
    for folder_idx in range(len(folders)):
        if dataset == 'fbp':
            generic_path = generic_f_fbps[folder_idx][:-1]+'/slice_fbp_####.tiff'
        elif dataset == 'tv':
            generic_path = generic_f_tvs[folder_idx][:-1]+'/slice_tv_####.tiff'

        A = np.arange(start_indices[folder_idx], end_indices[folder_idx])
        paths.append(ma.generate_paths(generic_path,A))
    return [item for sublist in paths for item in sublist]


def generate_batch_stitched_paths(dataset = 'fbp', folders = [1,2,3,4,5,6,7], start_indices = None, end_indices = None, n_batches = 4):
    generic_f_tv = '/dtu-compute/msaca/sliceA_xray_pc/output/tv_recon/r_###.'
    generic_f_tvs = ma.generate_paths(generic_f_tv,folders)

    generic_f_fbp ='/dtu-compute/msaca/sliceA_xray_pc/output/fbp_recon/r_###.'
    generic_f_fbps = ma.generate_paths(generic_f_fbp,folders)

    paths = []
    for folder_idx in range(len(folders)):
        if dataset == 'fbp':
            generic_path = generic_f_fbps[folder_idx][:-1]+'/slice_fbp_####.tiff'
        elif dataset == 'tv':
            generic_path = generic_f_tvs[folder_idx][:-1]+'/slice_tv_####.tiff'

        A = np.arange(start_indices[folder_idx], end_indices[folder_idx])
        paths.append(ma.generate_paths(generic_path,A))

    stitched_paths = []
    for j in range(len(paths)):
        N_slices = ma.find_largest_number('/dtu-compute/msaca/sliceA_xray_pc/output/cache/sino_007/sino_#####.tiff')
        subslices = np.arange(0,N_slices)
        base_size = len(subslices) // n_batches
        remainder = len(subslices) % n_batches
        split_arr = []
        start_idx = 0
        for i in range(n_batches):
            # For the first N-1 parts, add an extra element if there's a remainder
            end_idx = start_idx + base_size + (1 if i < remainder else 0)
            split_arr.append(subslices[start_idx:end_idx])
            start_idx = end_idx
        
        new_paths = []
        for path in paths[j]:
            number = int(path[-8:-5])  # Extracts the number as an integer
            for i, group in enumerate(split_arr, start=1):
                if number in group:
                    # Modify the path based on the group
                    new_path = path.replace('slice_tv_', f'slice_tv{i}_')
                    new_paths.append(new_path)
                    break
        stitched_paths.append(new_paths)

    stitched_paths = [item for sublist in stitched_paths for item in sublist]

    return stitched_paths

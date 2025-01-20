import tifffile
import importlib
import sys, os
import numpy as np
import module_auxiliary as ma

stich_indices(dataset = 'fbp'):
    folders = [1,2,3,4,5,6,7]
    d = 50
    D = 250
    dataset = 'fbp'
    generic_f_tv = '/dtu-compute/msaca/sliceA_xray_pc/output/tv_recon/r_###.'
    generic_f_tvs = ma.generate_paths(generic_f_tv,folders)

    generic_f_fbp ='/dtu-compute/msaca/sliceA_xray_pc/output/fbp_recon/r_###.'
    generic_f_fbps = ma.generate_paths(generic_f_fbp,folders)

    start_indices = []
    end_indices[]
    for folder_idx in range(len(folders)-1):
        if dataset == 'fbp':
            generic_path_pre = generic_f_fbps[folder_idx][:-1]+'/slice_fbp_####.tiff'
        elif dataset == 'tv':
            generic_path_pre = generic_f_tvs[folder_idx][:-1]+'/slice_tv_####.tiff'

        if dataset == 'fbp':
            generic_path_post = generic_f_fbps[folder_idx+1][:-1]+'/slice_fbp_####.tiff'
        elif dataset == 'tv':
            generic_path_post = generic_f_tvs[folder_idx+1][:-1]+'/slice_tv_####.tiff'
        
        print(generic_path_pre)
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

    return start_indices, end_indices



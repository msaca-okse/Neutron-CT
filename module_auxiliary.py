import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import minimize

def generate_paths(path, A):
    # Extract the folder and filename pattern
    folder, filename_pattern = os.path.split(path)
    
    # Determine the number of digits in the zero-padding
    num_digits = filename_pattern.count('#')
    filename_base = filename_pattern.replace('#' * num_digits, '{}')

    # Generate the list of paths
    paths = [
        os.path.join(folder, filename_base.format(str(num).zfill(num_digits)))
        for num in A
    ]
    
    return paths

def fits_loader(paths):
    fits_data = []
    for file in paths:
        with fits.open(file) as hdul:
            # Assume the data is in the primary HDU
            data = hdul[0].data
            fits_data.append(data)
    return np.stack(fits_data)


def determine_edge(sin1,bias = 0):
    # If sin1 is 2D: Sin1 is a sinogram of the form (angle, horizontal)
    # If sin1 is 3D. Sin1 is a collectoin of projections/sinograms of the form (vertical, angle, horizontal)
    # The function determines for each vertical layer, the index where the edge between the beam and no-beam is located. Output is size(sin1)[0] x 2.
    if len(np.shape(sin1))==2:
        logvars = np.log(np.var(sin1,axis=0))
        MIN = np.minimum(np.min(logvars,axis=1),-9)
        threshold = (logvars.max() - MIN)*0.5+MIN
        a = np.diff(logvars<threshold)
        index = np.where(a)[0]

        variances = np.empty(len(index))
        for i in range(len(index)):
            indexran = range(index[i] - 5, index[i]+5)
            variances[i] = np.var(logvars[indexran])

        top_two_indices = np.argsort(variances)[-2:][::-1]
        final_indices = np.sort(index[top_two_indices])
        out = np.abs([final_indices[0] + bias, final_indices[1]-bias])
    elif len(np.shape(sin1))==3:
        logvars =  np.log(np.var(sin1,axis=1)) # Take variance over angle axis for each pixel
        MIN = np.minimum(np.min(logvars,axis=1),-9)
        threshold = (np.max(logvars,axis=1) - MIN)*0.6+MIN
        a = np.diff(logvars<threshold[:,np.newaxis])
        out = np.empty((np.shape(sin1)[0],2))
        mask = np.zeros((np.shape(sin1)[0],np.shape(sin1)[2]))
        for j in range(np.shape(sin1)[0]):
            print(j)
            index = np.where(a[j,5:-5])[0]
            if len(index)==0:
                out[j,0] = 0
                out[j,1] = 1
            else:
                variances = np.empty(len(index))
                print(index)
                for i in range(len(index)):
                    indexran = range(index[i] - 5, index[i]+5)
                    variances[i] = np.var(logvars[j,indexran])
                top_two_indices = np.argsort(variances)[-2:][::-1]
                final_indices = np.sort(index[top_two_indices])
                out[j,:] = np.array([final_indices[0],final_indices[1]])
                print(out[j,:])
                mask[j,int(out[j,0]):int(out[j,1])] = 1
        mask = np.apply_along_axis(median_filter, axis=0, arr=mask, size=7)
        out = np.apply_along_axis(np.diff, axis=0, arr=mask)
        out = [np.where(row)[0] for row in out>0.5]

    else:
        out=0
        return out

# Note tat the problem with the 3d version is that there is no low variance for slices outside the beam. Solution is to ue x = min(min(variance), some predefined number),

def gaussian_mask(sigma=10,cutoff=3,edge_indices=100, N_pixels=1600):
    x1=np.array(range(-edge_indices[0],0))
    mask1 = np.exp(-x1**2/(2*sigma**2))
    x2 = np.array(range(N_pixels- edge_indices[1]))
    mask2 = np.exp(-1/(2*sigma**2)*x2**2)
    if len(mask1)>sigma*cutoff:
        mask1[:-sigma*cutoff] = 0
    if len(mask2)>sigma*cutoff:
        mask2[sigma*cutoff:] = 0
    return x1,x2,mask1,mask2

def circle_mask(shape, x0, y0, r):
    """Creates a binary mask of a circle with center (x0, y0) and radius r."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2
    return mask.astype(int)

def objective(params, array,slices=None):
    x0, y0, r = params
    mask = circle_mask(array.shape, x0, y0, r)
    if slices is not None:
        slices_m_1 = [num - 1 for num in slices]
        array_new = array[slices_m_1]
        mask_new = mask[slices_m_1]
        return np.sum((mask_new - array_new) ** 2)
    else:
        return np.sum((mask - array) ** 2)

def determine_edge2(sin1,bias = 0,slices=None):

    if slices is not None:
        slices_m_1 = [num - 1 for num in slices]
        sin1 = np.log(np.var(sin1,axis=1))
        A = np.zeros((np.max(slices),np.shape(sin1)[1]))
        A[slices_m_1] = sin1
        sin1 = A
    else:
        sin1 =  np.log(np.var(sin1,axis=1)) # Take variance over angle axis for each pixel

    MIN = np.minimum(np.min(sin1,axis=1),-9)
    threshold = (np.max(sin1,axis=1) - MIN)*0.6+MIN
    array = (sin1<threshold[:,np.newaxis])*1.0
    initial_guess = [np.shape(sin1)[1]//2, np.shape(sin1)[1]//2, np.shape(sin1)[1]//2]
    result = minimize(objective, initial_guess, args=(array,slices,), method='Nelder-Mead',
                bounds=[(0, array.shape[1]*2), (0, 2*array.shape[1]), (1, max(array.shape))],tol=1e-2)
    circle_pars = result.x
    mask = circle_mask(np.shape(array), circle_pars[0], circle_pars[1], circle_pars[2]-bias)
    if slices is not None:
        data = np.zeros(np.shape(mask))
        data[slices_m_1] = 1
    else:
        data = np.ones(np.shape(mask))

    mask = mask[slices_m_1] if slices is not None else mask
    out = abs(np.apply_along_axis(np.diff, arr=mask, axis=1))
    out_indices = [np.where(row)[0] for row in out>0.5]
    return mask, out_indices, data

def gaussian_padding(sin1,out_indices, sigma = 30, cutoff=4, pad_mean_window_size = 40):
    N_pixels = np.shape(sin1)[2]
    sin2 = sin1.copy()
    for i in range(len(out_indices)):
        out_index = out_indices[i]
        if len(out_index)==0:
            sin2[i,:] = 0
        elif len(out_index) == 2:
            x1,x2,mask1,mask2 = gaussian_mask(sigma=sigma,cutoff=cutoff,edge_indices=out_index,N_pixels=N_pixels)
            sin2[i,:,:out_index[0]] = np.mean(sin1[i,:,out_index[0]:out_index[0]+pad_mean_window_size],axis=1)[:,np.newaxis]*mask1[np.newaxis,:]
            sin2[i,:,out_index[1]:] = np.mean(sin1[i,:,out_index[1]-pad_mean_window_size:out_index[1]],axis=1)[:,np.newaxis]*mask2[np.newaxis,:]

    return sin2


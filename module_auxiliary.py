import os
import numpy as np
from astropy.io import fits

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



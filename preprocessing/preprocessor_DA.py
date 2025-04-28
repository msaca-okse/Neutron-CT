import time
import h5py
import os
import numpy as np

from integrator_DA import DA_loader_gpu
from multiprocessing import Pool




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

FOLDER_TO_WATCH = '/work3/msaca/sliceA_DA_cache'
input_files = 'scan-####_pilatus.h5'
A1 = np.arange(339, 449+1, 2)
A2 = np.arange(455, 471+1, 2)
A3 = np.arange(475, 487+1, 2)
A = np.concatenate([A1,A2,A3])

FILES_TO_ANALYZE = generate_paths(input_files, A)
ANALYZED_FILES = set()
output_path = '/dtu-compute/msaca/sliceA_diffraction/powder-crystal/integrated-####.h5'
OUTPUT_FILES = generate_paths(output_path, A)
batch_ids = range(181)  # Generate batch_id values 181


def is_file_fully_written(file_path, wait_time=1):
    """Check if a file is stable by monitoring its size."""
    initial_size = os.path.getsize(file_path)
    time.sleep(wait_time)
    return initial_size == os.path.getsize(file_path)




def analyze_file(file_path):
    """Read the h5 file, compute the sum, and save it to the specified output h5 file."""
    file_name = os.path.basename(file_path)
    ANALYZED_FILES.add(file_name)
    try:


        # Find the corresponding output filename
        if file_name in FILES_TO_ANALYZE:
            index = FILES_TO_ANALYZE.index(file_name)
            output_file = OUTPUT_FILES[index]
        else:
            print(f"Warning: {file_name} not found in FILES_TO_ANALYZE. Skipping.")
            return

        print('Reading the file ', file_path)


        with Pool(processes=12) as pool:
            results = pool.starmap(DA_loader_gpu, [(batch_id, file_path) for batch_id in batch_ids])

        #results = []
        #for batch_id in range(batch_ids):
        #    results.append(DA_loader_gpu(batch_id, file_path))    

        stacked_arrays = {key: [] for key in results[0].keys()}
        for result in results:
            for key in result:
                stacked_arrays[key].append(result[key])

        # Stack the arrays along axis=0
        for key in stacked_arrays:
            stacked_arrays[key] = np.stack(stacked_arrays[key], axis=0)


        with h5py.File(output_file, "w") as f_out:
            for key, array in stacked_arrays.items():
                f_out.create_dataset(key, data=array)


        print(f"Analyzed {file_path} saved to {output_file})")
        

    except Exception as e:
        print(f"Error reading {file_path}: {e}")



def process_file(file_path):
    """Check and process a file if it's in the list and not yet analyzed."""
    file_name = os.path.basename(file_path)
    if file_name in FILES_TO_ANALYZE and file_name not in ANALYZED_FILES:
        print(f"Detected {file_name}, checking stability...")
        if is_file_fully_written(file_path):
            analyze_file(file_path)


def check_existing_files():
    """Check for files already present in the folder at startup."""
    print("Checking for existing files...")
    for file_name in os.listdir(FOLDER_TO_WATCH):
        file_path = os.path.join(FOLDER_TO_WATCH, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file, not a folder
            process_file(file_path)


def check_for_new_files(directory, already_analyzed_files):
    """Periodically check for new files in the directory."""
    files_in_directory = set(os.listdir(directory))
    new_files = files_in_directory - already_analyzed_files
    if new_files:
        for new_file in new_files:
            if new_file in FILES_TO_ANALYZE:
                file_path = os.path.join(FOLDER_TO_WATCH, new_file)
                process_file(file_path)



def main():


    check_existing_files()  # Process already existing files at startup

    while True:
        print(ANALYZED_FILES)
        check_for_new_files(FOLDER_TO_WATCH, ANALYZED_FILES)
        time.sleep(30)  # Check every 30 seconds
        if set(FILES_TO_ANALYZE) == ANALYZED_FILES:
            break


if __name__ == "__main__":
    main()
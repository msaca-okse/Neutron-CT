import os
import module_auxiliary as ma
import numpy as np

folders = np.array([3,4])
generic_pc = '/dtu-compute/msaca/sliceA_xray_pc/output/cache/sino_###.'
generic_pcs = ma.generate_paths(generic_pc, folders)
for i in range(len(folders)):
    path_cache_sino = generic_pcs[i][:-1]
    def clear_folder(folder_path):
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Walk through the directory
            for root, dirs, files in os.walk(folder_path, topdown=False):
                # Remove files
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                # Remove subdirectories
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
                    print(f"Deleted folder: {dir_path}")
        else:
            print(f"The folder {folder_path} does not exist.")

    # Example usage
    clear_folder(path_cache_sino)
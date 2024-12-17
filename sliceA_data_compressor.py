import numpy as np
import imageio
from dxchange.reader import read_edf
import auxilliary_functions as af



def compress(data, factor = 65534/320000, clip = [0,65534]):
    data = data*factor
    data = np.clip(data, a_min = clip[0], a_max = clip[1])
    data = data.astype(np.uint16)
    return data


p_pre1 = '/dtu-compute/msaca/sliceA_xray_pc/HA900_3um_mars_rock_00'
p_pre2 = '_/HA900_3um_mars_rock_00'
q_pre1 = '/dtu-compute/msaca/sliceA_xray_pc/compressed_XA_00'
q_pre2 = '_/compressed_XA_00'

p_post1 = '_####.edf'
q_post1 = '_####.tiff'

p_post_dark = '_/dark.edf'
p_post_dark2 = '_/dark/darkend0000.tiff'
p_post_obeam = '_/obeam/refHST6000.tiff'

stride = 100
A = range(1,6001, stride)
angles = np.linspace(0,360,len(A))
folders = [1,2]
start_crop_z = 620
end_crop_z = 1410
start_crop_x = 300
end_crop_x = 2048
size_z = end_crop_z - start_crop_z
size_x = end_crop_x - start_crop_x
for i in range(len(folders)):
    edf_path_p = p_pre1 + str(folders[i]) + p_pre2 + str(folders[i]) + p_post1
    tiff_path_q = q_pre1 + str(folders[i]) + q_pre2 + str(folders[i]) + q_post1
    paths = af.generate_paths(edf_path_p, A)
    tiff_paths = af.generate_paths(tiff_path_q, A)
    for idx in range(len(paths)):
        image = read_edf(paths[idx])
        image = compress(image)
        image = np.squeeze(image)
        raw_data = image[start_crop_z:end_crop_z,start_crop_x:end_crop_x]
        imageio.imwrite(tiff_paths[idx], raw_data)

    edf_path_dark = p_pre1 + str(folders[i]) + p_post_dark
    edf_path_obeam = p_pre1 + str(folders[i]) + p_post_obeam

    tiff_path_dark = q_pre1 + str(folders[i]) + p_post_dark
    tiff_path_obeam = q_pre1 + str(folders[i]) + p_post_obeam


    image_dark = read_edf(edf_path_dark)
    image_obeam = read_edf(edf_path_obeam)

    image_dark = compress(image_dark)
    image_obeam = compress(image_obeam)

    image_dark = np.squeeze(image_dark)
    image_obeam = np.squeeze(image_obeam)
    raw_dark = image_dark[start_crop_z:end_crop_z,start_crop_x:end_crop_x]
    raw_obeam = image_obeam[start_crop_z:end_crop_z,start_crop_x:end_crop_x]

    imageio.imwrite(tiff_path_dark, raw_dark)
    imageio.imwrite(tiff_path_obeam, raw_obeam)
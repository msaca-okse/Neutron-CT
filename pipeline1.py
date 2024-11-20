import module_auxiliary as ma
from cil.io import TIFFStackReader
import sys
sys.path.append('/zhome/71/c/146676/Desktop/msaca/main/')
import numpy as np
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData
from cil.processors import CentreOfRotationCorrector
from cil.processors import RingRemover
from cil.plugins.astra import FBP
from cil.io import NEXUSDataWriter
from numba import cuda
import json

def pipeline(input_path,slices,output_path=None,angles = None):
    paths = ma.generate_paths(input_path, slices)
    reader = TIFFStackReader(paths)
    sinograms = reader.read()
    sinograms = (sinograms/2500-12)
    sinograms = sinograms[np.newaxis] if len(np.shape(sinograms)) == 2 else sinograms  # Add a dimension, if slice is only single slice

    mask, indices, data = ma.determine_edge2(sinograms,bias = 20,slices=slices)
    sinograms = ma.gaussian_padding(sinograms,indices, sigma = 30, cutoff=4, pad_mean_window_size = 50)

    # Define geometry
    N_pixels = np.shape(sinograms)[2]
    N_slices = len(slices)
    if angles is None:
        angles = np.linspace(0, 360, np.shape(sinograms)[1], endpoint=True, dtype=np.float32)

    ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_pixels//2+10,0])\
                                .set_angles(angles)\
                                .set_panel((N_pixels,N_slices), pixel_size=(1,1))\
                                .set_labels(labels=('vertical','angle','horizontal'))
    sinograms = AcquisitionData(sinograms, geometry=ag)
    ig = ag.get_ImageGeometry(resolution=1)

    # Center of rotatoin corrector
    target_value = 800
    index_target = min(range(len(slices)), key=lambda i: abs(slices[i] - target_value)) # get the index of slices that is closest to target value.
    processor = CentreOfRotationCorrector.xcorrelation(slice_index=index_target,ang_tol=0.2)
    processor.set_input(sinograms)
    sinograms = processor.get_output()

    # Ring removal corrector
    ringrmv = RingRemover(decNum=4, wname='db5', sigma=0.1, info=True)
    ringrmv.set_input(sinograms)
    sinograms = ringrmv.get_output()

    # FBP Reconstructor. Check if GPU is available
    if not cuda.is_available():
        raise ValueError("GPU is not available.")
    else:
        ag = sinograms.geometry
        device='gpu'
        fbp = FBP(ig,ag,device)
        reconstruction = fbp(sinograms)
        if output_path is not None:
            writer = NEXUSDataWriter(reconstruction, output_path)
            writer.write()
            meta_data = {'slices': slices, 'input_path': input_path}
            with open(output_path + '_metadata', 'w') as json_file:
                json.dump(meta_data, json_file)

        else:
            return reconstruction, sinograms
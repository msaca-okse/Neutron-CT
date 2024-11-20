import module_auxiliary as ma
from cil.io import TIFFStackReader
import sys
sys.path.append('/zhome/71/c/146676/Desktop/msaca/main/')
import numpy as np
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer
from cil.processors import CentreOfRotationCorrector
from cil.processors import RingRemover
from cil.io import NEXUSDataWriter
from numba import cuda
import json
from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, ISTA, PDHG, SPDHG
from cil.plugins.astra import FBP
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.utilities import callbacks
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator

def pipeline(input_path,slices,output_path=None,angles = None, iteration_history = 10,method='cgls',other_args = None):
    paths = ma.generate_paths(input_path, slices)
    reader = TIFFStackReader(paths)
    sinograms = reader.read()
    sinograms = (sinograms/2500-12)
    sinograms = sinograms[np.newaxis] if len(np.shape(sinograms)) == 2 else sinograms  # Add a dimension, if slice is only single slice

    mask, indices, data = ma.determine_edge2(sinograms,bias = 15,slices=slices)
    sinograms = ma.gaussian_padding(sinograms,indices, sigma = 30, cutoff=4, pad_mean_window_size = 50)
                    #    if isinstance(other_args, dict) and 'reconstruction_slices' in other_args:
                    #        reconstruction_slices = other_args['reconstruction_slices']
                    #        indices = [i for i, element in enumerate(slices) if element in reconstruction_slices]
                    #        sinograms = sinograms[indices]
                    #        slices = reconstruction_slices

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
        
        initial = ig.allocate(0)

        A = ProjectionOperator(ig, ag, device)

        if isinstance(other_args, dict) and 'return_data' in other_args and other_args['return_data']:
            data = {'ag': ag, 'ig':ig, 'sinograms': sinograms, 'projection_operator': A}
            return data
        
        if  isinstance(other_args, dict) and 'block_operator' in other_args and 'block_data' in other_args:
            block_operator = other_args['block_operator']
            block_data = other_args['block_data']
            print('Custom Block_data used')
        else:
            block_operator = A
            block_data = sinograms

        iteration_history = [iteration_history] if isinstance(iteration_history,int) else iteration_history
        reconstruction = []


################ FBP ####################
        if method == 'fbp':
            if isinstance(other_args, dict) and 'filter_type' in other_args:
                filter_type = other_args['filter_type']
            else:
                filter_type = 'ram-lak'
            sinograms.reorder('astra')
            device = 'gpu'
            fbp = FBP(ig,ag,device)
            reconstruction = fbp(sinograms)
            return reconstruction, sinograms

################ CGLS ###################
        elif method == 'cgls':
            reconstructor = CGLS(initial=initial, operator=block_operator, data=block_data, update_objective_interval = 5 )
            
################ SIRT ##########################
        elif method == 'sirt':
            constraint = IndicatorBox(lower=0)
            reconstructor = SIRT(initial=initial, operator=block_operator, data=block_data,update_objective_interval = 5, constraint=constraint)
            
################ GD ##############################
        elif method == 'gd':
            if isinstance(other_args, dict) and 'alpha' in other_args:
                alpha = other_args['alpha']

            if isinstance(other_args, dict) and 'loss' in other_args:
                loss = other_args['loss']
            else: 
                b = sinograms
                D = GradientOperator(ig)
                alpha = 10.0
                f1 = LeastSquares(A, b)
                f2 = OperatorCompositionFunction(L2NormSquared(),D)
                loss = f1 + (alpha**2)*f2
            reconstructor = GD(initial=initial, objective_function=loss, step_size=1/loss.L,update_objective_interval = 5)

#################### FISTA #########################
        elif method == 'fista':
            
            if isinstance(other_args, dict) and 'F_loss' in other_args and 'G_loss' in other_args:
                F = other_args['F_loss']
                G = other_args['G_loss']
            else: 
                b = sinograms
                F = LeastSquares(A, b)
                G = IndicatorBox(lower=0.0)
            reconstructor = FISTA(f=F, g=G, initial=initial, update_objective_interval = 5)

#################### ISTA #########################
        elif method == 'ista':
            if isinstance(other_args, dict) and 'F_loss' in other_args and 'G_loss' in other_args:
                F = other_args['F_loss']
                G = other_args['G_loss']
            else: 
                b = sinograms
                F = LeastSquares(A, b)
                G = IndicatorBox(lower=0.0)
                
            reconstructor = ISTA(f=F, g=G, initial=initial, update_objective_interval = 5)


###################### PDHG #############################
        elif method == 'pdhg':
            if isinstance(other_args, dict) and 'F_loss' in other_args and 'G_loss' in other_args and 'K_loss' in other_args:
                F = other_args['F_loss']
                G = other_args['G_loss']
                K = other_args['K_loss']
            else: 
                F = 0.5 * L2NormSquared(b=sinograms)
                alpha = 0.01
                G = alpha * L1Norm()
                K = A

            reconstructor = PDHG(f = F, g = G, operator = K, update_objective_interval = 10)

###################### SPDHG #############################
        elif method == 'spdhg':
            if isinstance(other_args, dict) and 'F_loss' in other_args and 'G_loss' in other_args and 'K_loss' in other_args:
                F = other_args['F_loss']
                G = other_args['G_loss']
                K = other_args['K_loss']
            else: 
                raise ValueError('Please specify F_loss, G_loss and K_loss options in other_args. See the SPDHG demo.')
                
            reconstructor = SPDHG(f = F, g = G, operator = K, update_objective_interval = 10)


## ## ## ## ## ## RECONSTRUCTION ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        prev = 0
        for i in range(len(iteration_history)):
            reconstructor.run(iteration_history[i] - prev, verbose=1)
            prev = iteration_history[i]
            reconstruction.append(reconstructor.solution.copy())



        if output_path is not None:
            writer = NEXUSDataWriter(reconstruction, output_path)
            writer.write()
            meta_data = {'slices': slices, 'input_path': input_path}
            with open(output_path + '_metadata', 'w') as json_file:
                json.dump(meta_data, json_file)

        else:
            return reconstruction, sinograms
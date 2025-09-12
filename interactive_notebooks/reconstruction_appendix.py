import sys
import os
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from cil.recon import FBP
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer
from cil.optimisation.functions import Function, SumFunction, IndicatorBox, WeightedL2NormSquared, TranslateFunction
import h5py
from cil.processors import Slicer
from cil.plugins.astra import ProjectionOperator
import xrayutilities as xu
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.interpolate import interp1d
import yaml
os.chdir("/zhome/71/c/146676/xrd_simulator/recon_material_decomp")



from cil.recon import FBP
#from cil.plugins.astra import FBP
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry, ImageData, BlockDataContainer
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.functions import L2NormSquared, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, TotalVariation, LeastSquares
from cil.optimisation.operators import BlockOperator, GradientOperator, IdentityOperator, FiniteDifferenceOperator
from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, ISTA, PDHG, SPDHG
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction, Function
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator



import yaml
import cv2
from scipy.ndimage import label, sum_labels
from cil.plugins.astra import ProjectionOperator
os.chdir("/zhome/71/c/146676/xrd_simulator/recon_material_decomp")
c0, c1 = 0, 256
alpha_dc_ = 0
pad = 0

def cgls1(data, angles, alpha_dc_, alpha=1, alpha_dx = 100, alpha_dy = 500, niter = 10):

    data_= zero_pad_3d(data, pad,value=np.mean(data[:,-10:]))
    y0, y1 = 0, data.shape[2]
    nchannels,ntheta, nx = np.shape(data)
    nchannels,ntheta, nx_ = np.shape(data_)

    ag = AcquisitionGeometry.create_Parallel2D(detector_position=[0,nx_//2])\
                                .set_angles(angles)\
                                .set_channels(nchannels)\
                                .set_panel((nx_), pixel_size=(1))\
                                .set_labels(['channel','angle', 'horizontal'])
    data_ = AcquisitionData(data_, geometry=ag)
    data_.reorder('astra')
    ig_ = ag.get_ImageGeometry()
    roi = {'horizontal_x':(pad,-pad,1), 'horizontal_y':(pad+y0,pad+y1,1)}
    processor = Slicer(roi)
    processor.set_input(ig_)
    ig = processor.get_output()
    device = 'gpu'
    fdx = FiniteDifferenceOperator(ig, direction='horizontal_x', bnd_cond='Neumann')
    fdy = FiniteDifferenceOperator(ig, direction='horizontal_y', bnd_cond='Neumann')
    fdc = FiniteDifferenceOperator(ig, direction='channel', bnd_cond='Neumann')
    factor = 0.03
    alpha_id = alpha*factor
    alpha_dx = alpha_dx*factor
    alpha_dy = alpha_dy*factor
    alpha_dc = alpha_dc_*factor
    L = alpha_id*IdentityOperator(ig)
    A = ProjectionOperator(ig, ag, 'gpu')
    FD = BlockOperator(alpha_dx * fdx,alpha_dy *  fdy, alpha_dc *  fdc)
    A_block = BlockOperator(L,FD, A)

    g_block = BlockDataContainer(L.range.allocate(0), FD.range.allocate(0), data_)
    # set up and run CGLS
    optimizer = CGLS(operator=A_block, data=g_block)
    optimizer.run(niter, verbose=1)
    recon_cgls = optimizer.solution.as_array().astype(np.float32)
    return recon_cgls


def sirt1(data, angles, alpha_id=1, niter = 10):

    data_= zero_pad_3d(data, pad,value=np.mean(data[:,-10:]))
    y0, y1 = 0, data.shape[2]
    nchannels,ntheta, nx = np.shape(data)
    nchannels,ntheta, nx_ = np.shape(data_)

    ag = AcquisitionGeometry.create_Parallel2D(detector_position=[0,nx_//2])\
                                .set_angles(angles)\
                                .set_channels(nchannels)\
                                .set_panel((nx_), pixel_size=(1))\
                                .set_labels(['channel','angle', 'horizontal'])
    data_ = AcquisitionData(data_, geometry=ag)
    data_.reorder('astra')
    ig_ = ag.get_ImageGeometry()
    roi = {'horizontal_x':(pad,-pad,1), 'horizontal_y':(pad+y0,pad+y1,1)}
    processor = Slicer(roi)
    processor.set_input(ig_)
    ig = processor.get_output()
    device = 'gpu'
    L = alpha_id*IdentityOperator(ig)
    A = ProjectionOperator(ig, ag, device)
    A_block = BlockOperator(L, A)
    g_block = BlockDataContainer(L.range.allocate(0), data_)
    # set up and run CGLS
    x0 = ig.allocate()
    constraint = IndicatorBox(lower=0)
    optimizer = SIRT(initial=x0, operator=A_block, data=g_block, constraint = constraint)
    optimizer.run(niter, verbose=1)
    recon_sirt = optimizer.solution.as_array().astype(np.float32)
    return recon_sirt



def fista1(data, angles, alpha_id=1, niter = 10):

    data_= zero_pad_3d(data, pad,value=np.mean(data[:,-10:]))
    y0, y1 = 0, data.shape[2]
    nchannels,ntheta, nx = np.shape(data)
    nchannels,ntheta, nx_ = np.shape(data_)

    ag = AcquisitionGeometry.create_Parallel2D(detector_position=[0,nx_//2])\
                                .set_angles(angles)\
                                .set_channels(nchannels)\
                                .set_panel((nx_), pixel_size=(1))\
                                .set_labels(['channel','angle', 'horizontal'])
    data_ = AcquisitionData(data_, geometry=ag)
    data_.reorder('astra')
    ig_ = ag.get_ImageGeometry()
    roi = {'horizontal_x':(pad,-pad,1), 'horizontal_y':(pad+y0,pad+y1,1)}
    processor = Slicer(roi)
    processor.set_input(ig_)
    ig = processor.get_output()
    device = 'gpu'
    L = alpha_id*IdentityOperator(ig)
    A = ProjectionOperator(ig, ag, device)
    A_block = BlockOperator(L, A)
    g_block = BlockDataContainer(L.range.allocate(0), data_)
    # set up and run CGLS
    x0 = ig.allocate()
    constraint = IndicatorBox(lower=0)
    optimizer = FISTA(f = F, g=G, initial=x0)
    optimizer.run(niter, verbose=1)
    recon_sirt = optimizer.solution.as_array().astype(np.float32)
    return recon_sirt



def zero_pad_3d(array: np.ndarray, N: int,value=0) -> np.ndarray:
    """
    Pads a 3D NumPy array with zeros along the last two dimensions by N pixels.
    The first dimension (axis=0) remains unchanged.

    Parameters:
    array (np.ndarray): Input 3D array.
    N (int): Number of pixels to pad along the last two dimensions.

    Returns:
    np.ndarray: Zero-padded 3D array.
    """
    if N < 0:
        raise ValueError("Padding size N must be non-negative")
    
    pad_width = ((0, 0), (0, 0), (N, N))  # No padding on axis=0, N on other axes
    return np.pad(array, pad_width, mode='constant', constant_values=0)



class Huber(Function):
    """
    f(u) = lam * sum_i rho_delta(u_i), with
    rho_delta(r) = 0.5 r^2                if |r| <= delta
                 = delta(|r| - 0.5 delta) otherwise
    """
    def __init__(self, delta=1.0, lam=1.0):
        super().__init__()
        self.delta = float(delta)
        self.lam = float(lam)

    def __call__(self, x):
        u = x.as_array()
        a = np.abs(u); d = self.delta
        val = np.where(a <= d, 0.5 * u*u, d*(a - 0.5*d))
        return self.lam * float(np.sum(val))

    # prox_{tau f}(y)
    def proximal(self, y, tau, out=None):
        d, lam = self.delta, self.lam
        t = tau * lam
        u = y.as_array()
        a = np.abs(u)

        res = np.empty_like(u)
        mask_q = a <= (1.0 + t) * d
        res[mask_q]  = u[mask_q] / (1.0 + t)
        res[~mask_q] = np.sign(u[~mask_q]) * np.maximum(a[~mask_q] - t*d, 0.0)

        if out is None:
            z = y.copy(); z.fill(res); return z
        else:
            out.fill(res); return out

    # prox_{sigma f*}(y)
    def proximal_conjugate(self, y, sigma, out=None):
        lam, d = self.lam, self.delta
        u = y.as_array()
        p = u / (1.0 + sigma / lam)            # quadratic shrink
        p = np.clip(p, -lam*d, lam*d)          # project onto |p|<=lam*delta

        if out is None:
            z = y.copy(); z.fill(p); return z
        else:
            out.fill(p); return out

    def convex_conjugate(self, y):
        lam, d = self.lam, self.delta
        p = y.as_array()
        if np.any(np.abs(p) > lam*d):
            return np.inf
        return float(0.5 * np.sum((p*p) / lam))
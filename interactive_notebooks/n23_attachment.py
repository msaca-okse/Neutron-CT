import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
os.chdir('/zhome/71/c/146676/main/')
from helpers import module_auxiliary as ma
from loaders import stitcher_XA
from loaders import loader_XA_to_NA
import SimpleITK as sitk
import copy

def get_attenuation_coefficients():
    # -*- coding: utf-8 -*-

    """

    Created on Fri Dec 13 10:37:39 2024



    @author: ebna

    """



    import matplotlib.pyplot as plt

    import numpy as np



    #%% X-ray attenution as calculated in matlab



    # theoretical attenuation lengths for minerals at 73 keV

    att_xr = [0.5262, 0.5337, 0.5882, 0.6421, 0.6578, 0.6679, 

            0.7205, 0.8334, 0.9474, 1.2057, 1.5400, 1.6905, 

            2.1636, 2.2703, 2.4571, 2.5057, 2.9957, 3.0109, 

            5.6032, 10.6152, 0.5262, 2.9957] # [1/cm]



    labels = ['Plg_alb', 'PthF_Na', 'PthF_K', 'Plg_anor', 'Oli_fost', 'Opx_enst', 

            'Spin', 'Clpx_CaMg', 'Chlap', 'Clpx_CaFeAl', 'Rut', 'Opx_fer', 

            'Oli_fay', 'Ilm', 'Chr', 'Pyr', 'Mag', 'Hem', 

            'Zir', 'Badd']


    #%% Theoretical neutron attenuation length

    elements = ["H", "O", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti", "Cr", "Fe", "Zr"]

    Z = [1, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 24, 26, 40]

    N_A = 6.022e23



    # Density from NIST table of material constants: https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html

    rho = np.array([8.375e-5, 1.332e-3, 0.971, 1.740, 2.699, 2.330, 2.200, 2.000, 2.995e-3, 0.862, 1.550, 4.540, 7.180, 7.874, 6.506])  # [g/cm^3]

    M = np.array([1.008, 15.999, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.098, 40.078, 47.867, 51.996, 55.845, 91.224]) # Molar mass [g/mol]

    # Neutron total cross sections, from NIST https://ncnr.nist.gov/resources/n-lengths/

    sigma = np.array([82.35, 4.232, 3.81, 3.773, 1.734, 2.338, 3.484, 1.556, 50.3, 4.06, 3.26, 10.44, 6.54, 14.18, 6.645]) *1e-24 # [cm^2]



    # Calculate attenuations for minerals

    plg_alb  = sigma[2]*N_A*rho[2]/(M[2]) + 2*sigma[4]*N_A*rho[4]/(M[4]) + 2*sigma[5]*N_A*rho[5]/(M[5]) + 8*sigma[1]*N_A*rho[1]/(M[1]) # NaAl2Si2O8

    pthf_Na  = sigma[2]*N_A*rho[2]/(M[2]) + sigma[4]*N_A*rho[4]/(M[4]) + 3*sigma[5]*N_A*rho[5]/(M[5]) + 8*sigma[1]*N_A*rho[1]/(M[1]) # NaAlSi3O8

    pthf_K   = sigma[9]*N_A*rho[9]/(M[9]) + sigma[4]*N_A*rho[4]/(M[4]) + 3*sigma[5]*N_A*rho[5]/(M[5]) + 8*sigma[1]*N_A*rho[1]/(M[1]) # KAlSi3O8

    plg_anor = sigma[10]*N_A*rho[10]/(M[10]) + 2*sigma[4]*N_A*rho[4]/(M[4]*N_A) + 2*sigma[5]*N_A*rho[5]/(M[5]) + 8*sigma[1]*N_A*rho[1]/(M[1]) # CaAl2Si2O8

    oli_fost = 2*sigma[3]*N_A*rho[3]/(M[3]) + sigma[5]*N_A*rho[5]/(M[5]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # Mg2SiO4

    opx_enst = 2*sigma[3]*N_A*rho[3]/(M[3]) + 2*sigma[5]*N_A*rho[5]/(M[5]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # Mg2Si2O4



    spin        = sigma[3]*N_A*rho[3]/(M[3]) + 2*sigma[4]*N_A*rho[4]/(M[4]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # MgAl2O4

    clpx_CaMg   = sigma[10]*N_A*rho[10]/(M[10]) + sigma[3]*N_A*rho[3]/(M[3]) + 2*sigma[5]*N_A*rho[5]/(M[5]) + 6*sigma[1]*N_A*rho[1]/(M[1]) # CaMgSi2O6

    chlap       = 5*sigma[10]*N_A*rho[10]/(M[10]) + 3*sigma[6]*N_A*rho[6]/(M[6]) +12*sigma[1]*N_A*rho[1]/(M[1]) + sigma[8]*N_A*rho[8]/(M[8]) # Ca5P3O12Cl

    clpx_CaFeAl = sigma[10]*N_A*rho[10]/(M[10]) + sigma[13]*N_A*rho[13]/(M[13]) + 2*sigma[4]*N_A*rho[4]/(M[4]) + 6*sigma[1]*N_A*rho[1]/(M[1]) # CaFeAl2O6

    rut         = sigma[11]*N_A*rho[11]/(M[11]) + 2*sigma[1]*N_A*rho[1]/(M[1])  # TiO2 

    opx_fer     = 2*sigma[13]*N_A*rho[13]/(M[13]) + 2*sigma[5]*N_A*rho[5]/(M[5]) + 6*sigma[1]*N_A*rho[1]/(M[1])# Fe2Si2O6



    oli_fay = 2*sigma[13]*N_A*rho[13]/(M[13]) + sigma[5]*N_A*rho[5]/(M[5]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # Fe2SiO4

    ilm     = sigma[13]*N_A*rho[13]/(M[13]) + sigma[11]*N_A*rho[11]/(M[11]) + 3*sigma[1]*N_A*rho[1]/(M[1]) # FeTiO3

    chrom   = sigma[13]*N_A*rho[13]/(M[13]) + 2*sigma[12]*N_A*rho[12]/(M[12]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # FeCr2O4

    pyr     = sigma[13]*N_A*rho[13]/(M[13]) + 2*sigma[7]*N_A*rho[7]/(M[7]) # FeS2

    mag     = 3*sigma[13]*N_A*rho[13]/(M[13]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # Fe3O4

    hem     = 2*sigma[13]*N_A*rho[13]/(M[13]) + 3*sigma[1]*N_A*rho[1]/(M[1]) # Fe2O3 



    zir = sigma[-1]*N_A*rho[-1]/(M[-1]) +sigma[5]*N_A*rho[5]/(M[5]) + 4*sigma[1]*N_A*rho[1]/(M[1]) # ZrSiO4

    badd = sigma[-1]*N_A*rho[-1]/(M[-1]) + 2*sigma[1]*N_A*rho[1]/(M[1]) # ZrO2



    H = sigma[0]*N_A*1/(M[0])



    att_ne = [plg_alb, pthf_Na, pthf_K, plg_anor, oli_fost, opx_enst, 

            spin, clpx_CaMg, chlap, clpx_CaFeAl, rut, opx_fer, 

            oli_fay, ilm, chrom, pyr, mag, hem, 

            zir, badd, H]


    #%% mass attenuation

    # Weight fractions calculated based on number of atoms in unit cell.

    # Except for perthitic feldspar and clinopyroxene, where they are weighted from chemical composition



    pthf_Na  = 1/13*sigma[2]*N_A/(M[2]) + 1/13*sigma[4]*N_A/(M[4]) + 3/13*sigma[5]*N_A/(M[5]) + 8/13*sigma[1]*N_A/(M[1]) # NaAlSi3O8 #  Basically the same as albite

    pthf_K   = 1/13*sigma[9]*N_A/(M[9]) + 1/13*sigma[4]*N_A/(M[4]) + 3/13*sigma[5]*N_A/(M[5]) + 8/13*sigma[1]*N_A/(M[1]) # KAlSi3O8

    clpx_CaMg   = 1/10*sigma[10]*N_A/(M[10]) + 1/10*sigma[3]*N_A/(M[3]) + 2/10*sigma[5]*N_A/(M[5]) + 6/10*sigma[1]*N_A/(M[1]) # CaMgSi2O6

    clpx_CaFeAl = 1/10*sigma[10]*N_A/(M[10]) + 1/10*sigma[13]*N_A/(M[13]) + 2/10*sigma[4]*N_A/(M[4]) + 6/10*sigma[1]*N_A/(M[1]) # CaFeAl2O6



    w_alb = M[2] + 4*M[4] + 4*M[5] + 8*M[1]

    plg_alb  = M[2]/w_alb* sigma[2]*N_A/(M[2]) + 4*M[4]/w_alb* sigma[4]*N_A/(M[4]) + 4*M[5]/w_alb* sigma[5]*N_A/(M[5]) + 8*M[1]/w_alb* sigma[1]*N_A/(M[1]) # NaAl2Si2O8

    w_anor   = 4*M[10] + 8*M[4] + 8*M[5] + 33*M[1]

    plg_anor = 4*M[10]/w_anor* sigma[10]*N_A/(M[10]) + 8*M[4]/w_anor* sigma[4]*N_A/(M[4]*N_A) + 8*M[5]/w_anor* sigma[5]*N_A/(M[5]) + 32*M[1]/w_anor* sigma[1]*N_A/(M[1]) # CaAl2Si2O8

    w_fors   = 2*M[3] + M[5] + 3*M[1]

    oli_fost = 2*M[3]/w_fors* sigma[3]*N_A/(M[3]) + M[5]/w_fors* sigma[5]*N_A/(M[5]) + 3*M[1]/w_fors* sigma[1]*N_A/(M[1]) # Mg2SiO4

    w_enst   = 2*M[3] + 2*M[5] + 6*M[1]

    opx_enst = 2*M[3]/w_enst*sigma[3]*N_A/(M[3]) + 2*M[5]/w_enst*sigma[5]*N_A/(M[5]) + 6*M[1]/w_enst*sigma[1]*N_A/(M[1]) # Mg2Si2O4



    spin        = M[3]/(M[3]+M[4]+M[1])* sigma[3]*N_A/(M[3]) + M[4]/(M[3]+M[4]+M[1])* sigma[4]*N_A/(M[4]) + M[1]/(M[3]+M[4]+M[1])* sigma[1]*N_A/(M[1]) # MgAl2O4

    w_chlap = (2*M[10]+M[6]+3*M[1]+M[8])

    chlap       = 2*M[10]/w_chlap* sigma[10]*N_A/(M[10]) + M[6]/w_chlap* sigma[6]*N_A/(M[6]) + 3*M[1]/w_chlap* sigma[1]*N_A/(M[1]) + M[8]/w_chlap* sigma[8]*N_A/(M[8]) # Ca5P3O12Cl

    rut         = M[11]/(M[11]+M[1])* sigma[11]*N_A/(M[11]) + M[1]/(M[11]+M[1])* sigma[1]*N_A/(M[1])  # TiO2 

    w_fer = 2*M[13] + 2*M[5] + 6*M[1]

    opx_fer     = 2*M[13]/w_fer* sigma[13]*N_A/(M[13]) + 2*M[5]/w_fer* sigma[5]*N_A/(M[5]) + 6*M[1]/w_fer* sigma[1]*N_A*rho[1]/(M[1])# Fe2Si2O6



    oli_fay = 2*M[13]/(2*M[13]+M[5]+3*M[1])* sigma[13]*N_A/(M[13]) + M[5]/(2*M[13]+M[5]+3*M[1])* sigma[5]*N_A/(M[5]) + 3*M[1]/(2*M[13]+M[5]+3*M[1])* sigma[1]*N_A/(M[1]) # Fe2SiO4

    ilm     = M[13]/(M[13]+M[11]+M[1])* sigma[13]*N_A/(M[13]) + M[11]/(M[13]+M[11]+M[1])* sigma[11]*N_A/(M[11]) + M[1]/(M[13]+M[11]+M[1])* sigma[1]*N_A/(M[1]) # FeTiO3

    chrom   = M[13]/(M[13]+M[12]+M[1])* sigma[13]*N_A/(M[13]) + M[12]/(M[13]+M[12]+M[1])* sigma[12]*N_A/(M[12]) + M[1]/(M[13]+M[12]+M[1])* sigma[1]*N_A/(M[1]) # FeCr2O4

    pyr     = M[13]/(M[13]+M[7])* sigma[13]*N_A/(M[13]) + M[7]/(M[13]+M[7])* sigma[7]*N_A/(M[7]) # FeS2

    mag     = 2*M[13]/(2*M[13]+M[1])* sigma[13]*N_A/(M[13]) + M[1]/(2*M[13]+M[1])* sigma[1]*N_A/(M[1]) # Fe3O4

    hem     = M[13]/(M[13]+M[1])* sigma[13]*N_A/(M[13]) + M[1]/(M[13]+M[1])* sigma[1]*N_A/(M[1]) # Fe2O3 



    zir = (M[-1]/(M[-1]+M[5]+M[1]))* sigma[-1]*N_A/(M[-1]) + (M[5]/(M[-1]+M[5]+M[1])) * sigma[5]*N_A/(M[5]) + (M[1]/(M[-1]+M[5]+M[1])) * sigma[1]*N_A/M[1] # ZrSiO4

    badd = M[-1]/(M[-1]+2*M[1])* sigma[-1]*N_A/(M[-1]) + 2*M[1]/(M[-1]+2*M[1])* sigma[1]*N_A/(M[1]) # ZrO2



    H = sigma[0]*N_A/(M[0])

    OH = M[0]/(M[0]+M[1]) * sigma[0]*N_A/M[0] + M[1]/(M[0]+M[1]) * sigma[1]*N_A/M[1]

    FeOOH = M[0]/(M[0]+M[1]+M[13]) * sigma[0]*N_A/M[0] + M[1]/(M[0]+M[1]+M[13]) * sigma[1]*N_A/M[1] + M[13]/(M[0]+M[1]+M[13]) * sigma[13]*N_A/M[13]



    att_mass_ne = [plg_alb, pthf_Na, pthf_K, plg_anor, oli_fost, opx_enst, 

                spin, clpx_CaMg, chlap, clpx_CaFeAl, rut, opx_fer, 

                oli_fay, ilm, chrom, pyr, mag, hem, 

                zir, badd, plg_alb, mag]

    att_xr = [v * 0.15 for v in att_xr]
    att_mass_ne = [v * 2 for v in att_mass_ne]


    labels =     [
        'Albite',         # Plg_alb → Na(AlSi3O8)_plagioclase_albite
        'PthF_Na',        # No match
        'PthF_K',         # No match
        'Plg_anor',       # No match
        'Forsterite',     # Oli_fost → Mg2SiO4_olivine_forsterite
        'Enstatite',      # Opx_enst → Mg2Si2O6_pyroxene_enstatite
        'Spinel',         # Spin → MgAl2O4_spinel
        'Diopside',       # Clpx_CaMg → CaMgSi2O6_clinopyroxene_diopsite
        'Chlorapatite',   # Chlap → Ca5(PO4)3Cl_chlorapatite
        'Clpx_CaFeAl',    # No match
        'Rutile',         # Rut → TiO2_rutile
        'Ferrosilite',    # Opx_fer → Fe2Si2O6_pyroxene_ferrosilite
        'Oli_fay',        # No match
        'Ilmenite',       # Ilm → FeTiO3_ilmenite
        'Magnetite',       # Chr → FeCr2O4_chromite (not Magnetite)
        'Pyrite',            # No exact match (could be Pyrite or Pyrrhotite)
        'Magnetite',            # No exact match (possibly Magnetite or Titanomagnetite)
        'Hematite',       # Hem → Fe2O3_hematite
        'Zircon',         # Zir → ZrSiO4_zircon
        'Baddeleyite',     # Badd → ZrO2_baddeleyite
        'Orthoclase',
        'Wüstite'
    ]

    ## add numbers for orthoclase and wüstite



    return np.array(att_xr), np.array(att_mass_ne), labels
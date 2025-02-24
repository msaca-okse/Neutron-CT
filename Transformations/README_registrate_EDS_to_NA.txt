the transformation
"transform_EDS_to_NA.tfm" registers an xray volume loaded by
XA_surf_volume_ = np.load('/dtu-compute/msaca/cache/XA_surf1.npy'),

the index of the surface can be extracted by
edge = ma.extract_edge(NA, axis=1),
with NA being the full Neutron volume.

The EDS volume should be loaded by

path = '/dtu-compute/msaca/sliceA_eds/EDS_BB_A_raw_files/edsMg.tiff'
eds = tifffile.imread(path)
eds[eds>150] = 0
eds = np.fliplr(eds)
#eds = np.flipud(eds)
eds = eds[4000::2,::2]/100
XA_surf = XA_surf_volume_[:,35]

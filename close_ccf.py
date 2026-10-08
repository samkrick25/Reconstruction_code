# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:40:11 2026
binarize ccf volume
@author: samkr
"""

from reconstructions.utils.filedirs import allen_ccf_10um
import nibabel as nib
from nibabel.spatialimages import SpatialImage
from scipy.ndimage import binary_fill_holes
import numpy as np
from tqdm import tqdm

allen_ccf = nib.load(allen_ccf_10um)
allen_ccf_data = np.asanyarray(allen_ccf.dataobj)

bin_ccf = (allen_ccf_data > 0.5).astype(int)


output_ccf = np.zeros_like(bin_ccf)

for i in tqdm(range(allen_ccf_data.shape[0]), desc='filling holes in ccf'):
    binary_fill_holes(bin_ccf[i], output=output_ccf[i])

#%% 
from nibabel.loadsave import save
sf = r'D:\allen_brain_atlas\2025_release\ccf_10um_closed.nii'

img = SpatialImage(output_ccf.astype(np.uint16), affine=None)

save(img, sf)
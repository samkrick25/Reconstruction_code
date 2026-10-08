# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:43:18 2026
ccf stuff
@author: samkr
"""

from reconstructions.utils.filedirs import allen_ccf_10um, bin_ccf, ccf_boundary
from brainglobe_atlasapi import BrainGlobeAtlas
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

#2025_boundary = r"D:\allen_brain_atlas\2025_release\annotation_boundary_10.nii"

coords = [13165.083252, 5787.026584, 4761.346117]
tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()

ccf_annot = np.asanyarray(nib.load(allen_ccf_10um).dataobj)

print('loading BG atlas')
bg_atlas = BrainGlobeAtlas(atlas_name='allen_mouse_10um', check_latest=False)
print('fetching annotation volume')
bg_annot = bg_atlas.annotation.astype(np.uint8)

#plt.imshow(bounds[1317])
#plt.plot(tenmicron[1], tenmicron[2], marker='*')

def check_all_array_vals(arr1, arr2):
    shape1 = arr1.shape
    shape2 = arr2.shape
    
    if shape1 != shape2:
        print('Arrays are different sizes')
        return
    
    boolval = (arr1==arr2).all()
    return boolval

check = check_all_array_vals(bg_annot, ccf_annot)
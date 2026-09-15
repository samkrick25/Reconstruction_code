# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:26:00 2026
visualize genotypes
@author: samkr
"""

from brainrender import Scene, settings
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.cellLists import cells_by_geno, cells_by_pheno
from reconstructions.utils.filedirs import celldir
from reconstructions.utils import cameras
import os
from tqdm import tqdm

#create a mapping from cell name to pheno identity
def map_val_to_key(d):
    nd = {}
    for k, v in d.items():
        for val in v:
            nd[val]=k
    return nd

cellMap = map_val_to_key(cells_by_pheno)

#would be more informative to color cells by their phenotype rather than their genotype tbh
gColors = {'AiP1999AiE2255': 'red', 'AiP1996AiE2199': 'pink', 'AiP1836AiE2256': 'maroon', 'calb2': 'cyan',
           'cart':'blue', 'crh': 'navy', 'dbh': 'orange', 'gal': 'seagreen', 'grp': 'olive', 'ntrk1': 'lightgreen',
           'sim1': 'yellow', 'som': 'orchid', 'vgat': 'sienna', 'vglut1': 'mediumpurple', 'vglut2': 'purple',
           'AAVwt': 'magenta', 'RVwt': 'slategrey', 'ALMantero': 'cornflowerblue'}

pColors = {'sensory': 'red', 'premotor': 'cyan', 'mossy': 'green', 'GRN': 'purple', 
          'sensorimotor': 'pink', 'vestibular': 'yellow', 'forebrain': 'orange'}

scene = Scene(atlas_name = 'allen_mouse_10um')
settings.SHOW_AXES = False
settings.INTERACTIVE = False
settings.OFFSCREEN = True
# =============================================================================
# settings.BACKGROUND_COLOR = 'black'
# settings.ROOT_ALPHA = 0.075
# =============================================================================


root = scene.get_actors()[0]
root._needs_silhouette=False

savedir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\genos\top'


for geno, cells in tqdm(cells_by_geno.items(), desc='Taking screenshots'):
    savef = os.path.join(savedir, geno+'_top.png')
    for cell in tqdm(cells, desc='Adding cells to scene:'):
        cf = os.path.join(celldir,cell+'.swc')
        actors = pp.swap_for_brainrender(cf, dendrite='black', soma='black', axon=pColors[cellMap[cell]])
        for actor in actors:
            scene.add(actor)
    
    scene.screenshot(savef, scale=12, camera=cameras.rootcam)
    scene.close()
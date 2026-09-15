# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:41:08 2026
visualize cell phenotypes
@author: samkr
"""
from reconstructions.utils.filedirs import celldir
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.cellLists import cells_by_pheno
from reconstructions.utils import cameras as c
from brainrender import Scene, settings
import os
from tqdm import tqdm

savefile = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\phenos\mirrorsoma_sag.png'

settings.INTERACTIVE = False
settings.OFFSCREEN = True
settings.BACKGROUND_COLOR = 'black'
settings.ROOT_ALPHA = 0.05

zoomsagcam = dict(
    pos=(13035.2, 2056.86, 28708.8),
    focal_point=(10821.8, 6046.66, -8825.85),
    viewup=(4.60553e-3, -0.994359, -0.105969),
    roll=179.379,
    distance=37810.9,
    clipping_range=(30242.9, 46267.7),
)
zoomcorcam = dict(
    pos=(36452.3, 5748.27, -5524.06),
    focal_point=(5604.26, 6329.36, -5546.72),
    viewup=(0, -1.00000, 0),
    roll=180.000,
    distance=30853.5,
    clipping_range=(16237.4, 46997.4),
)
zoomrootcam = dict(
    pos=(11175.6, -31652.3, -5851.92),
    focal_point=(11245.4, 4288.82, -5603.42),
    viewup=(-1.00000, 0, 0),
    roll=74.3239,
    distance=35942.0,
    clipping_range=(27647.7, 45453.5),
)

ccf_scene = Scene(atlas_name = 'allen_mouse_10um')

root = ccf_scene.get_actors()[0]
root._needs_silhouette = False

# =============================================================================
# ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.075)
# ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.075, color='pink')
# =============================================================================

regs = ['PSV', 'SPVI', 'SPVC', 'SPVO', 'NTS'    ]
for reg in regs:
    ccf_scene.add_brain_region(reg, color='red', alpha=0.2, silhouette=False)

medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,1))
ccf_scene.slice(plane=medplane)

#ccf_scene.render(camera=c.topcam)

colors = {'sensory': 'red', 'premotor': 'cyan', 'mossy': 'green', 'GRN': 'purple', 'sensorimotor': 'pink', 'vestibular': 'yellow', 'forebrain': 'orange'}
#render axons
for file in tqdm(os.listdir(celldir), desc='Adding neurons to scene'):
    cellname = file.split('.')[0]
    if cellname in cells_by_pheno['sensory']:
        #continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['sensory'], 
                                         skip_dendrite = True, neurite_radius = 5, soma_radius=0, alpha=0.7)
    if cellname in cells_by_pheno['sensorimotor']:
        continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['sensorimotor'], 
                                         skip_dendrite = True, neurite_radius = 5, soma_radius=0)
    if cellname in cells_by_pheno['premotor']:
        continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['premotor'], 
                                         skip_dendrite = True, neurite_radius = 5, soma_radius=0)
    if cellname in cells_by_pheno['GRN']:
        continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['GRN'], 
                                         skip_dendrite = True, neurite_radius = 5, soma_radius=0)
    if cellname in cells_by_pheno['forebrain']:
        continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['forebrain'], 
                                         skip_dendrite = True, neurite_radius = 5, soma_radius=0)
    if cellname in cells_by_pheno['mossy']:
        continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['mossy'], 
                                         skip_dendrite = True, neurite_radius = 5, soma_radius=0)
    if cellname in cells_by_pheno['vestibular']:
        continue
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['vestibular'], 
                                         dendrite='white', soma='white', neurite_radius = 5, soma_radius=0)
    
    for actor in actors:
        ccf_scene.add(actor)

sf = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\phenos\sens\sens_w_reg_sag.png'
ccf_scene.screenshot(sf, camera=c.sagcam, scale=6)
#ccf_scene.render(camera=c.corcam)

#render somas

# =============================================================================
# for file in tqdm(os.listdir(celldir), desc='Adding somas to scene'):
#     cellname = file.split('.')[0]
#     if cellname in cl.sensory:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['sensory'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     if cellname in cl.sensorimotor:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['sensorimotor'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     if cellname in cl.premotor:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['premotor'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     if cellname in cl.GRN:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['GRN'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     if cellname in cl.forebrain:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['forebrain'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     if cellname in cl.mossy:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['mossy'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     if cellname in cl.vestibular:
#         #continue
#         actors = pp.swap_for_brainrender(os.path.join(celldir, file), soma=colors['vestibular'], 
#                                          skip_dendrite = True, skip_axon=True, soma_radius=30, mirror_soma=True)
#     
#     for actor in actors:
#         ccf_scene.add(actor)
# =============================================================================
  
        
#ccf_scene.render()
  
#ccf_scene.screenshot(savefile, scale=6, camera=zoomsagcam)

# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:28:17 2026

@author: samkr
"""

from brainrender import Scene
from brainrender import settings
from reconstructions.utils import preprocess_funcs as pf
from reconstructions.utils import cameras
import numpy as np
from reconstructions.utils import visualize as vis
from reconstructions.utils.cellLists import cells_by_pheno, pColors
from reconstructions.utils.filedirs import celldir, annotated

#create a mapping from cell name to pheno identity
def map_val_to_key(d):
    nd = {}
    for k, v in d.items():
        for val in v:
            nd[val]=k
    return nd

cells_to_pheno = map_val_to_key(cells_by_pheno)

#premotors for u19
#AA0947, AA1196, AL0006, N023-703070, N034-674191-IB, N071-709222

#path to swc to be visualized
#n1 = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N013-703070.swc"
#n2 = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\AA0947.swc"

#list of cells to be visualized
celllist = ['N056-709222', 'N011-763259-KS', 'N086-686955-HD', 'N090-686955-LS',
'AL0006', 'N066-709222', 'N058-709222', 'N071-709222', 'N064-709222', 
'N022-706301-KA', 'N031-674185-FMR', 'N039-674185-IB', 'N090-708369-VM']
#['N058-709222', 'N012-763259-JG', 'N022-674185-HD', 'N049-709222', 'N090-708369-VM', 'N015-763259-JM', 'N002-703070'] save for sensory cluster?

#edit brainrender settings
settings.SHOW_AXES=None
settings.INTERACTIVE=False
settings.OFFSCREEN=True
settings.ROOT_COLOR='white'
settings.ROOT_ALPHA=0.075
settings.BACKGROUND_COLOR='black'


#set up brainrender scene
ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True, check_latest=False)
root = ccf_scene.get_actors()[0]
root._needs_silhouette = False
print('scene set')

# =============================================================================
# regions = ['Acs5']
# copies = [ccf_scene.atlas.get_region(r).mesh.clone() for r in regions]
# cut = pf.get_mesh_onehem(ccf_scene, mesh=copies, hem='left')
# verts = [m.vertices for m in cut]
# =============================================================================

#add desired brain regions
ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.1)
ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.1, color='pink')
#ccf_scene.add_brain_region('LING', silhouette=False, color='orange', alpha=0.2)
#ccf_scene.add_brain_region('Acs5', silhouette=False, color='purple', alpha=0.2)
# =============================================================================
#ccf_scene.add_brain_region('V', silhouette=False, color='green', alpha=0.2)
# =============================================================================
#ccf_scene.add_brain_region('RM', silhouette=False, color='purple', alpha=0.2)
# ccf_scene.add_brain_region('VII', silhouette=False, color='blue', alpha=0.2)
# =============================================================================
#ccf_scene.add_brain_region('PSV', silhouette=False, color='red', alpha=0.2)
#ccf_scene.add_brain_region('SPVO', silhouette=False, color='cyan', alpha=0.2)
#ccf_scene.add_brain_region('SPVI', silhouette=False, color='red', alpha=0.2)
# =============================================================================

#get cell to color mapping
color_map = {}
for cell in celllist:
    pheno = cells_to_pheno[cell]
    color = pColors[pheno]
    color_map[cell] = color

sf = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\morphometrics_w_projs\classification\cells_wrong_gt50pct.png'

folder = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\phox2bcells'

#render cells
vis.render_cells(ccf_scene, camera=cameras.rootcam, celllist=celllist, celldir=annotated, savef=sf, screenshot=True, scale=6, soma_radius=20, color_map=color_map,
                 skip_axon=False)
    
#aa1331 soma coords 11663.7333265 6014.9173821875 6910.73133900001 
#aa1332 soma coords 11676.512623375 5906.1368678125 7027.4413675 


#ccf_scene.screenshot(sf, camera=cameras.rootcam, scale=12)
#ccf_scene.render(camera=cameras.MYtopcam)

'''
useful brainrender shortcuts when you have a scene open

    i     print info about the last clicked object     
    I     print color of the pixel under the mouse     
    Y     show the pipeline for this object as a graph 
    <- -> use arrows to reduce/increase opacity        
    x     toggle mesh visibility                       
    w     toggle wireframe/surface style               
    l     toggle surface edges visibility              
    p/P   hide surface faces and show only points      
    1-3   cycle surface color (2=light, 3=dark)        
    4     cycle color map (press shift-4 to go back)   
    5-6   cycle point-cell arrays (shift to go back)   
    7-8   cycle background and gradient color          
    09+-  cycle axes styles (on keypad, or press +/-)  
    k     cycle available lighting styles              
    K     toggle shading as flat or phong              
    A     toggle anti-aliasing                         
    D     toggle depth-peeling (for transparencies)    
    U     toggle perspective/parallel projection       
    o/O   toggle extra light to scene and rotate it    
    a     toggle interaction to Actor Mode             
    n     toggle surface normals                       
    r     reset camera position                        
    R     reset camera to the closest orthogonal view  
    .     fly camera to the last clicked point         
    C     print the current camera parameters state    
    X     invoke a cutter widget tool                  
    S     save a screenshot of the current scene       
    E/F   export 3D scene to numpy file or X3D         
    q     return control to python script              
    Esc   abort execution and exit python kernel  
'''
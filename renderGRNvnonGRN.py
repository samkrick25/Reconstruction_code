# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:40:58 2026

@author: samkr
"""
import vtk
vtk.vtkMultiThreader.SetGlobalMaximumNumberOfThreads(1)


from brainrender import Scene, settings
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import cameras
import pickle
from tqdm import tqdm
import os

THRESH = 2

mossys = ['AA0922', 'AA1263', 'N010-651324', 'N013-703070', 'N016-715345-HD', 'N017-703070', 'N017-715345-YV', 'N022-703070',
          'N023-715346-PC', 'N024-715345-SA', 'N030-651895', 'N031-651895', 'N031-715345-DS', 'N035-674191-FMR', 'N037-674185-IB',
          'N038-674185', 'N041-674191-AR', 'N044-674191-SP', 'N056-686955-JN', 'N057-686955-SA', 'N113-708369-JN', 
          'N114-708369-HS', 'N115-708369-BP']

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
frequencies = frequencies.drop('N004-674185-DS', axis=0)
merged = pp.merge_regions(frequencies)

celldir = r"reconstructions\data\IRNPARN_cells\swcsfromjson"
savedir = r'images'
rootcam = dict(
    pos=(6861.42, -108506, -5802.38),
    focal_point=(7080.29, 4292.90, -5022.47),
    viewup=(-1.00000, 0, 0),
    roll=74.3246,
    distance=112801,
    clipping_range=(103739, 123466),
)
cam2 = dict(
    pos=(7807.52, -7155.86, -5773.68),
    focal_point=(7829.74, 4296.07, -5694.50),
    viewup=(-1.00000, 0, 0),
    roll=74.3239,
    distance=11452.2,
    clipping_range=(2162.03, 24963.9),
)

settings.SHOW_AXES=False
settings.ROOT_COLOR=[0.8,0.8,0.8]
settings.OFFSCREEN=True
ccf_scene = Scene(atlas_name='allen_mouse_10um')
root = ccf_scene.get_actors()[0]
root._needs_silhouette=False

regions = ['GRN', 'MRN']
copies = [ccf_scene.atlas.get_region(r).mesh.clone() for r in regions]
cut = pp.get_mesh_onehem(ccf_scene, mesh=copies, hem='left')
verts = [m.vertices for m in cut]
# =============================================================================
# ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
# ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
# =============================================================================
ccf_scene.add_brain_region('GRN', silhouette=False, alpha=0.2, color='blue')
ccf_scene.add_brain_region('MRN', silhouette=False, alpha=0.2, color='orange')
#ccf_scene.add_brain_region('XII', silhouette=False, alpha=0.2, color='green')

masked = merged.astype(bool)
MRNGRN = merged.loc[(merged['GRN'] != 0) | (merged['MRN'] != 0)]

LUT = {'MRN':(255,140,0),'GRNMRN':(198,78,198),'GRN':(0,0,200)}

actors_by_key = {'MRN': [], 'GRN': [], 'GRNMRN': []}


for cell, vals in tqdm(MRNGRN.iterrows(), desc='Loading neurons'):
    print(f'Processing: {cell}', flush=True)
    fn = os.path.join(celldir, cell+'.swc')
    if vals['GRN'] > 0 and vals['MRN'] > 0:
        key = 'GRN'
        alpha = 0.484
    elif vals['GRN'] > 0 and vals['MRN'] == 0:
        key='GRN'
        alpha=0.603
        continue
    elif vals['MRN'] > 0 and vals['GRN'] == 0:
        key='MRN'
        alpha=0.601
    else:
        continue
    #print(f"[BEFORE BUILD] {cell} — key={key}", flush=True)
    if key == 'GRN':
        actors = pp.swap_for_brainrender(fn, axon=LUT[key], skip_dendrite=True, soma=LUT[key], alpha=1, neurite_radius=5, soma_radius=5, res=12, mesh=verts)
    if key == 'MRN':
        actors = pp.swap_for_brainrender(fn, axon=LUT[key], skip_dendrite=True, soma=LUT[key], alpha=1, neurite_radius=15, soma_radius=5, res=12, mesh=verts)
    #print(f"[AFTER BUILD] {cell} — {len(actors)} actors", flush=True)
    for i, actor in enumerate(actors):
        #print(f"[BEFORE ADD] {cell} actor {i}/{len(actors)}", flush=True)
        ccf_scene.add(actor)
        #print(f"[AFTER ADD] {cell} actor {i}/{len(actors)}", flush=True)

# =============================================================================
#     actors_by_key[key].extend(actors)
#     
# for key, actors in actors_by_key.items():
#     if not actors:
#         continue
#     merged_actor = merge(actors)   # one VTK object per color group
#     #merged_actor.alpha(0.484 if key == 'GRNMRN' else 0.603)
#     ccf_scene.add(merged_actor)
# =============================================================================
# =============================================================================
# GRNcells = []
# nonGRNcells = []
# 
# for cell, targets in merged.iterrows():
#     if targets['GRN'] > THRESH:
#         GRNcells.append(cell)
#     if targets['GRN'] <= THRESH:
#         nonGRNcells.append(cell)
#         
# for file in tqdm(os.listdir(celldir), desc='Loading neurons'):
#     cellname = file.split('.')[0]
#     filepath = os.path.join(celldir, file)
#     if cellname in mossys:
#         continue
#     if cellname in GRNcells:
#         actors = pp.swap_for_brainrender(filepath, axon='green', skip_dendrite=True, soma='green', neurite_radius=8, soma_radius=4)
#         for actor in actors:
#             ccf_scene.add(actor)
#     if cellname in nonGRNcells:
#         actors = pp.swap_for_brainrender(filepath, axon='blue', skip_dendrite=True, soma='blue', neurite_radius=8, soma_radius=4)
#         for actor in actors:
#             ccf_scene.add(actor)
# =============================================================================


#ccf_scene.render(camera=cameras.topcam)
ccf_scene.screenshot(name=savedir+'\\'+'GRNvnonGRNtop6.png', camera=rootcam, scale=3)
ccf_scene.close()

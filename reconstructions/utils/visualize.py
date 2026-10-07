# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:23:38 2026
code to visualize IRN/PARN cells with brainrender
@author: economolab
"""

from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils.filedirs import somaspkl, allcoordswapped, parcellated_neurons
import reconstructions.utils.cameras as cams
from brainrender.actors import Points
import pickle
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from vedo import Tube, Sphere, Line
import vtk
from vedo import Mesh
from scipy.spatial import Delaunay

MIDLINEZ = 5750
MIDLINEZ_10UM = 570

# =============================================================================
# #get meshes for IRN/PARN from brainrender Scene
# scene_for_bounds = Scene(atlas_name='allen_mouse_10um', root=False)
# scene_for_bounds.add_brain_region('IRN')
# scene_for_bounds.add_brain_region('PARN')
# actors = scene_for_bounds.get_actors()
# IRNmesh = actors[0]
# IRNvertices = IRNmesh.mesh.vertices
# PARNmesh = actors[1]
# PARNvertices = PARNmesh.mesh.vertices
# 
# #find anterior bound and set antIRN/PARN posterior bound
# IRNap = [vertex[0] for vertex in IRNvertices]
# PARNap = [vertex[0] for vertex in PARNvertices]
# IRNPARNap = IRNap+PARNap
# MedRNa_bound = np.min(IRNPARNap)
# MedRNp_bound = np.max(IRNPARNap)
# MedRNrange = MedRNp_bound-MedRNa_bound
# MedRNmid = MedRNa_bound + 800
# 
# # =============================================================================
# # #dorsal ventral separation (NOT FUNCTIONAL YET)
# # IRNdv = [vertex[1] for vertex in IRNvertices]
# # PARNdv = [vertex[1] for vertex in PARNvertices]
# # RNdv = IRNdv+PARNdv
# # dorsbound = np.min(RNdv)
# # ventbound = np.max(RNdv)
# # med = np.mean([int(dorsbound), int(ventbound)])
# # 
# # =============================================================================
# #get lists of antIRN/PARN vs postIRN/PARN cells
# somas = pickle.load(open(somaspkl, 'rb'))
# antIPARNcells = []
# postIPARNcells = []
# for cell, soma in somas.items():
#     if MedRNa_bound <= soma['x'] < MedRNmid:
#         antIPARNcells.append(cell)
#     if MedRNmid <= soma['x'] <= MedRNp_bound:
#         postIPARNcells.append(cell)
#         
# 
# # =============================================================================
# # somas = pickle.load(open(somaspkl, 'rb'))
# # antIPARNcells = []
# # postIPARNcells = []
# # for cell, soma in somas.items():
# #     if dorsbound <= soma['x'] < med:
# #         antIPARNcells.append(cell)
# #     if med <= soma['x'] <= ventbound:
# #         postIPARNcells.append(cell)
# # =============================================================================
#         
#         
# #get endpoints of IRN/PARN cells split by their A->P position
# antIPARNends = {}
# postIPARNends = {}
# for file in tqdm(os.listdir(parcellated_neurons), desc='Loading endpoints'):
#     cellname = file.split('.')[0]

## I GOT RID OF GET_ENDPOINTS_FROM_FILE_PARCELLATED SO THIS LINE IS BROKEN
#REWRITE AT SOME POINT
#     endpoints = ld.get_endpoints_from_file_parcellated(os.path.join(parcellated_neurons, file))
#     if cellname in antIPARNcells:
#         antIPARNends[cellname] = endpoints
#     if cellname in postIPARNcells:
#         postIPARNends[cellname] = endpoints
# =============================================================================

#this function is also broke while above code doesn't work, i just want this file to be rendering stuff
# =============================================================================
# def show_endpoints(regions, root, colors, alphas, camera=cams.sagcam, unilat=False, ontlevel='structure'):
#     '''
#     visualize endpoints of IRN/PARN cells in any region defined by allen ccf 10um v3 
#     (will rewrite to use other resolutions, binary expansion of regions and visualization of those)
# 
#     Parameters
#     ----------
#     *regions : str
#         str of regions to be rendered and endpoints to be selected for visualization
#     root : bool
#         to be passed to brainrender.Scene.render, bool to render wholebrain mesh or not
#     colors: list
#         list of colors for each region to be rendered as, colors and regions must be the same length
#     alphas: list
#         list of alpha values for each region
#     camera: reconstructions.utils.cameras variable
#         what camera to use when rendering Scene, default is saggital view
#     
#     
# 
#     Returns
#     -------
#     None.
# 
#     '''
# 
#     if len(regions) != len(colors):
#         raise ValueError('region and color list must be of same length!')
#     if len(regions) != len(alphas):
#         raise ValueError('region and alpha list must be of same length!')
#         
#     ccf_scene = Scene(atlas_name='allen_mouse_10um', root=root)
#     for region, color, alpha in zip(regions, colors, alphas):
#         ccf_scene.add_brain_region(region, color=color, alpha=alpha, silhouette=False)
#     
#     #this is messing up, returning empty lists
#     antIPARNinreg = pp.get_nodes_in_region(antIPARNends, regions, parcellated=True, ontlevel=ontlevel, kind='by_cell')
#     postIPARNinreg = pp.get_nodes_in_region(postIPARNends, regions, parcellated=True, ontlevel=ontlevel, kind='by_cell')
#     
#     antIPARNXIIswapped = pp.tenmicron_to_one(antIPARNinreg, allcoordswapped)
#     postIPARNXIIswapped = pp.tenmicron_to_one(postIPARNinreg, allcoordswapped)
#     #finish cleaning up from here later, need to rewrite this a little since above is now by cell, first need to look at coordswapped for actual coords
#     antIPARNXIIcoords = []
#     for cell, nodes in antIPARNXIIswapped.items():
#         if unilat:
#             coords = pp.get_coords(nodes, mirror=True)
#         else:
#             coords = pp.get_coords(nodes, mirror=False)
#         if coords.any():
#             antIPARNXIIcoords.append(coords)
#     antIPARNXIIflat = np.array([node for cell in antIPARNXIIcoords for node in cell])
# 
#     postIPARNXIIcoords = []
#     for cell, nodes in postIPARNXIIswapped.items():
#         if unilat:
#             coords = pp.get_coords(nodes, mirror=True)
#         else:
#             coords = pp.get_coords(nodes, mirror=False)
#         if coords.any():
#             postIPARNXIIcoords.append(coords)
#     postIPARNXIIflat = np.array([node for cell in postIPARNXIIcoords for node in cell])
#     
#     antPoints = Points(antIPARNXIIflat, colors='red', radius=10)
#     postPoints = Points(postIPARNXIIflat, colors='blue', radius=10)
#     ccf_scene.add(antPoints)
#     ccf_scene.add(postPoints)
#     
#     if unilat:
#         medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
#         ccf_scene.slice(plane=medplane)
#     
#     ccf_scene.render(camera=camera)
#     
#     return antIPARNXIIswapped, postIPARNXIIswapped
# =============================================================================

def get_inmesh_mask(swc_df, mesh):
    """
    Return a boolean numpy array (len == len(swc_df)) where True means the
    node falls inside at least one of the provided mesh(es).

    Applies midline reflection for contralateral nodes so a single
    ipsilateral mesh can match bilateral projections to the same region.
    If mesh is empty, all nodes are True (no filtering).
    """
    mesh_list = _normalize_meshes(mesh)
    if not mesh_list:
        return np.ones(len(swc_df), dtype=bool)

    coords = swc_df[['x', 'y', 'z']].values.copy()   # copy — never touch swc_df
    over = coords[:, 2] > MIDLINEZ
    coords[over, 2] = 2 * MIDLINEZ - coords[over, 2]  # vectorized reflection

    mask = np.zeros(len(coords), dtype=bool)
    for verts in mesh_list:
        tri = Delaunay(verts)
        mask |= tri.find_simplex(coords) >= 0
        if mask.all():          # early exit once every node is accounted for
            break
    return mask

def _normalize_meshes(mesh):
    """
    Normalize the mesh argument into a flat list of (N,3) vertex arrays.
      - np.array([])               → []   (no filtering)
      - single (N,3) ndarray       → [mesh]
      - list of (N,3) ndarrays     → mesh (empty arrays dropped)
    """
    if isinstance(mesh, np.ndarray):
        return [] if mesh.size == 0 else [mesh]
    if isinstance(mesh, list):
        return [m for m in mesh if isinstance(m, np.ndarray) and m.size > 0]
    return []


def _mesh_has_data(mesh):
    """Return True when mesh contains at least one non-empty vertex array."""
    return len(_normalize_meshes(mesh)) > 0


def swc_to_line_actors(swc_df, skip_dendrite=False, skip_axon=False, axon_color='blue', dendrite_color='red',
                       soma_color='black', neurite_radius=4, soma_radius=15, alpha=1, res=12,
                       include_soma=True, inmesh_ids=None, mirror_soma=False):
    """
    Parameters
    ----------
    inmesh_ids : set[int] | None
        Pre-computed set of SWC node IDs whose CCF-space coordinates fell
        inside at least one region mesh.  A section is rendered only if at
        least one of its node IDs appears in this set.
        None → no spatial filtering (render everything).

        Filtering at section level means entire sections are kept or
        discarded — no mid-section breaks, so crossing-midline paths
        are never split into two disconnected pieces.
    """
    nodes = swc_df.set_index('id')

    # ── Build children from the FULL, unfiltered node set ────────────────────
    # Never prune nodes here; intact topology is what prevents mid-path breaks.
    children = defaultdict(list)
    for node_id, row in nodes.iterrows():
        if row['parent'] != -1:
            if skip_dendrite and row['type'] == 3:
                continue
            children[int(row['parent'])].append(node_id)

    soma_rows = nodes[nodes['type'] == 1]
    soma_id   = int(soma_rows.iloc[0].name) if len(soma_rows) > 0 else -1

    if include_soma and len(soma_rows) > 0:
        soma_pos = soma_rows.iloc[0][['x', 'y', 'z']].tolist()
        if mirror_soma:
            x, _, _ = soma_pos
            if x>MIDLINEZ:
                diff = x-MIDLINEZ
                x = MIDLINEZ - diff
                soma_pos[0] = x
        actors = [Sphere(pos=soma_pos, r=soma_radius, c=soma_color, alpha=alpha)]
        if skip_axon and skip_dendrite:
            return actors
    else:
        actors = []

    sections_by_type = defaultdict(list)

    # soma_id check is always included regardless of include_soma so that
    # direct soma children are correctly identified as section starts.
    section_starts = [
        node_id for node_id, row in nodes.iterrows()
        if row['type'] != 1
        and (row['parent'] == -1
             or int(row['parent']) == soma_id
             or len(children[int(row['parent'])]) > 1)
    ]

    for start_id in section_starts:
        section_ids = []
        parent_id   = int(nodes.loc[start_id, 'parent'])
        if parent_id != -1 and parent_id != soma_id and parent_id in nodes.index:
            section_ids.append(parent_id)

        current_id = start_id
        while True:
            section_ids.append(current_id)
            kids = children[current_id]
            if len(kids) == 1:
                current_id = kids[0]
            else:
                break

        if len(section_ids) < 2:
            continue

        # ── Section-level mesh filter ─────────────────────────────────────────
        # Keep entire section if ANY of its nodes were inside a mesh region.
        # Using set intersection is O(min(|inmesh_ids|, |section_ids|)).
        if inmesh_ids is not None and not inmesh_ids.intersection(section_ids):
            continue

        pts   = nodes.loc[section_ids, ['x', 'y', 'z']].values
        ntype = int(nodes.loc[start_id, 'type'])
        sections_by_type[ntype].append(pts)

    type_color_map = {2: axon_color, 3: dendrite_color}

    for ntype, sections in sections_by_type.items():
        if skip_dendrite and ntype == 3:
            continue
        if skip_axon and ntype==2:
            continue
        color = type_color_map.get(ntype, axon_color)
        actor = build_tube_actor_from_sections(
            sections, radius=neurite_radius, color=color, alpha=alpha, res=res
        )
        if actor is not None:
            actors.append(actor)

    return actors


def create_cell_actors(swcpath, axon='green', dendrite='black', soma='black',
                         skip_dendrite=False, skip_axon=False, neurite_radius=4, soma_radius=15,
                         alpha=1, res=12, mesh=np.array([]), mirror_soma=False):
    swc_df = pd.read_csv(
        swcpath, comment='#', sep=r'\s+',
        names=['id', 'type', 'x', 'y', 'z', 'r', 'parent']
    )

    has_mesh = _mesh_has_data(mesh)

    # ── Step 1: in-mesh check in CCF space (BEFORE axis swap) ────────────────
    # Mesh vertices are in CCF space; swc_df is also still in CCF space here.
    # Doing the check now means the coordinate frames always match.
    if has_mesh:
        inmesh_mask = get_inmesh_mask(swc_df, mesh)          # boolean ndarray
        inmesh_ids  = set(swc_df['id'][inmesh_mask].values)  # set of SWC node IDs
    else:
        inmesh_ids = None

    # ── Step 2: axis swap for brainrender display ─────────────────────────────
    swc_df['x'], swc_df['z'] = swc_df['z'].copy(), swc_df['x'].copy()

    # ── Step 3: build actors from full topology, section-level filter ─────────
    cell_actors = swc_to_line_actors(
        swc_df,
        axon_color=axon, dendrite_color=dendrite, soma_color=soma,
        skip_dendrite=skip_dendrite, skip_axon=skip_axon, neurite_radius=neurite_radius,
        soma_radius=soma_radius, alpha=alpha, res=res,
        include_soma=not has_mesh,
        inmesh_ids=inmesh_ids, mirror_soma=mirror_soma
    )
    return cell_actors

def build_tube_actor_from_sections(sections, radius, color, alpha, res):
    """
    Takes a list of point arrays (one per morphological section) and builds
    them all into a single vtkPolyData, then applies ONE vtkTubeFilter to it.
    Returns a single vedo Mesh actor.

    This avoids creating one vtkTubeFilter per section, which causes VTK
    internal state corruption after enough instantiations.
    """
    points = vtk.vtkPoints()
    lines  = vtk.vtkCellArray()
    pt_offset = 0

    for pts in sections:
        if len(pts) < 2:
            continue

        # Guard against degenerate/NaN coordinates
        pts = np.array(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            continue

        # Remove consecutive duplicate points (zero-length segments crash VTK)
        keep = np.ones(len(pts), dtype=bool)
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[i-1]) < 1e-6:
                keep[i] = False
        pts = pts[keep]
        if len(pts) < 2:
            continue

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(pts))
        for i, pt in enumerate(pts):
            points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
            polyline.GetPointIds().SetId(i, pt_offset + i)
        lines.InsertNextCell(polyline)
        pt_offset += len(pts)

    if pt_offset == 0:
        return None

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polydata)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(res)
    tube.CappingOn()
    tube.Update()

    # Normalize color to 0-1 range if passed as 0-255 tuple
    if isinstance(color, (tuple, list)) and any(c > 1 for c in color):
        color = tuple(c / 255.0 for c in color)

    return Mesh(tube.GetOutput(), c=color, alpha=alpha)


def get_mesh_onehem(scene, mesh, hem='left'):
    root_mesh = scene.atlas.get_region('root').mesh
    if scene.atlas.metadata['symmetric']:
        mesh_center = (root_mesh.bounds().reshape((3,2)).mean(axis=1))
    else:
        mesh_center = root_mesh.center_of_mass()
    
    normal = (0, 0, 1) if hem == "right" else (0, 0, -1)
    plane = scene.atlas.get_plane(pos=mesh_center, norm=normal)
    if not isinstance(mesh, list):
        #this is cutting the whole mesh instead of just giving one hem, look into it in a moment
        mesh.cut_with_plane(origin=plane.center, normal=plane.normal)
        mesh.cap()
    else:
        for m in mesh:
            m.cut_with_plane(origin=plane.center, normal=plane.normal)
            m.cap()
    return mesh

def render_cells(celllist, celldir, scene, camera, screenshot=False, savef=None, scale=None, color_map=None, **kwargs):
    '''
    render cells in brainrender. Pass in a scene with any desired region meshes
    already added, any visualization settings already set, this will just
    draw cells and either render the scene or take a screenshot

    Parameters
    ----------
    celllist : list
        list of cell names to be rendered.
    celldir : str
        Path to directory containing cell swc files
    scene : brainrender Scene
        brainrender Scene object to draw the cells in.
    camera : dict
        dictionary containing parameters for brainrender's camera.
    screenshot: bool
        switch to enable screenshots, True to take a screenshot and save to path specified in savef
    savedir: str
        Path to file to save a screenshot to
    scale: int
        Desired scale for brainrender to take a screenshot at
    color_dict: dict
        mapping for cell keys to color to be rendered, a dictionary where keys are cell names in celllist and 
        values are the desired color for that cell to be rendered as
    kwargs: dict
        keyword arguments to be passed to swap_for_brainrender

    Returns
    -------
    None.

    '''
    for cell in tqdm(os.listdir(celldir), desc='rendering cells'):
        
        cellname = cell.split('.')[0]
        fn = os.path.join(celldir, cell)
        
        if cellname in celllist:
            kwargs.update({'axon': color_map[cellname]})
            actors = create_cell_actors(fn, **kwargs)
            
            for actor in actors:
                scene.add(actor)
                
    if screenshot:
        scene.screenshot(savef, camera=camera, scale=scale)
        scene.close()
    
    else:
        scene.render(camera=camera)
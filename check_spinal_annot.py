# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:20:43 2026
verify my spinal cord annotation
@author: samkr
"""

from reconstructions.utils.filedirs import celldir, bin_ccf
from reconstructions.utils.load_data import check_if_spinal_cord
from brainrender import Scene, settings
from tqdm import tqdm
import nibabel as nib
from collections import defaultdict
from vedo import Sphere, Mesh
import pandas as pd
import numpy as np
import vtk
from scipy.spatial import Delaunay

MIDLINEZ = 5750
MIDLINEZ_10UM = 570

bin_ccf_img = nib.load(bin_ccf)
bin_ccf_data = bin_ccf_img.dataobj


def swc_to_line_actors(swc_df, skip_dendrite=False, skip_axon=False, axon_color='blue', dendrite_color='red',
                       soma_color='black', neurite_radius=4, soma_radius=15, alpha=1, res=12,
                       include_soma=True, inmesh_ids=None, mirror_soma=False, filter_spinal=False, spinal_color=None):
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
    count=0
    ids = []
    for node_id, row in tqdm(nodes.iterrows(), desc='Parsing nodes'):
        if row['parent'] != -1:
            if skip_dendrite and row['type'] == 3:
                continue
            if filter_spinal:
                ccfcoords = [row['z'], row['y'], row['x']]
                #tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()
                spinal_check, kind = check_if_spinal_cord(ccfcoords, bin_ccf_data)
                if spinal_check:
                    nodes.at[node_id, 'type'] = 4
                    count+=1
                    ids.append(node_id)
            if row['type'] == 5 or row['type'] == 6:
                nodes.at[node_id, 'type'] = nodes.loc[int(row['parent'])]['type']
            children[int(row['parent'])].append(node_id)
    
    if filter_spinal:
        print(count)
        
    soma_rows = nodes[nodes['type'] == 1]
    soma_id   = int(soma_rows.iloc[0].name) if len(soma_rows) > 0 else -1 #this check was written for multiple soma nodes, which shouldn't exist

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
        if row['type'] != 1 #not soma
        and (row['parent'] == -1 #parent is soma
             or int(row['parent']) == soma_id #parent is soma
             or len(children[int(row['parent'])]) > 1) #has multiple children
             or row['type'] == 4 #nodes in spinal cord always count as starts
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

    type_color_map = {2: axon_color, 3: dendrite_color, 4: spinal_color}

    for ntype, sections in sections_by_type.items():
        if skip_dendrite and ntype == 3:
            continue
        if skip_axon and ntype==2:
            continue
        color = type_color_map.get(ntype)
        actor = build_tube_actor_from_sections(
            sections, radius=neurite_radius, color=color, alpha=alpha, res=res
        )
        if actor is not None:
            actors.append(actor)

    return actors

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

def create_cell_actors(swcpath, axon='green', dendrite='black', soma='black',
                         skip_dendrite=False, skip_axon=False, neurite_radius=4, soma_radius=15,
                         alpha=1, res=12, mesh=np.array([]), mirror_soma=False, filter_spinal=False, spinal_color=None):
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
        inmesh_ids=inmesh_ids, mirror_soma=mirror_soma, filter_spinal=filter_spinal, spinal_color=spinal_color
    )
    return cell_actors

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

swc = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N015-703070.swc"

actors = create_cell_actors(swc, skip_axon=False, filter_spinal=True, spinal_color='red')

settings.INTERACTIVE = True
settings.OFFSCREEN = False

scene = Scene(atlas_name='allen_mouse_10um')

for actor in actors:
    scene.add(actor)
    
scene.render()
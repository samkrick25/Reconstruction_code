# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:23:38 2026
code to visualize Allen Institute swcs with brainrender/vedo
@author: samkr
"""

from reconstructions.utils.filedirs import parcellation_mappkl, allen_ccf_10um
import pickle
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from vedo import Sphere
import vtk
from vedo import Mesh
import nibabel as nib


MIDLINEZ = 5750
MIDLINEZ_10UM = 570

allen_ccf = nib.load(allen_ccf_10um)
allen_ccf_data = np.asanyarray(allen_ccf.dataobj)

parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))


#INTERNAL FUNCTIONS
def _get_parcel_id_from_reg(regabv):
    '''
    find allen parcellation id from region abbreviation

    Parameters
    ----------
    regabv : str
        Allen CCF region abbreviation to find parcellation index for.

    Returns
    -------
    int. Parcellation index associated with the given region in Allen CCF

    '''
    
    return parcellation_map.loc[parcellation_map['parcellation_term_acronym']==regabv]['parcellation_index']


def _swc_to_line_actors(swc_df, skip_dendrite=False, skip_axon=False, axon_color='blue', dendrite_color='red',
                       soma_color='black', neurite_radius=4, soma_radius=15, alpha=1, res=12,
                       include_soma=True, regs_to_render=None, mirror_soma=False):
    """
    generate Actor objects from an swc file that has been read into a pd.DataFrame. this basically draws cells as
    consecutive dots at a given 
    
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

    #find parcellation ids for desired regions to render, will filter nodes based on this
    if regs_to_render:
        parcel_ids = [_get_parcel_id_from_reg(abv).iloc[0].astype(np.float64) for abv in regs_to_render]

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
    #write in a check to add every node that has a different region annot than its parent to be a section start
    if not regs_to_render:     
        section_starts = [
            node_id for node_id, row in nodes.iterrows()
            if row['type'] != 1
            and (row['parent'] == -1
                 or int(row['parent']) == soma_id
                 or len(children[int(row['parent'])]) > 1)
        
        ]
    
    #so this might cause an issue for the branch/endpoints of dendrites? unsure how it will manifest
    #but here is where to check if there's weirdness, idk if this will add a section start if the dendrite is in the regions i want to render
    #which i guess shouldn't be an issue as long as you don't try to render dendrites and axon length in a given region
    if regs_to_render:
        section_starts = [
            node_id for node_id, row in nodes.iterrows()
            if row['type'] != 1
            and row['type'] != 3
            and row['parcelID'] in parcel_ids 
        ]

    #add children 
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
        actor = _build_tube_actor_from_sections(
            sections, radius=neurite_radius, color=color, alpha=alpha, res=res, parcelIDs=parcel_ids if regs_to_render else None
        )
        if actor is not None:
            actors.append(actor)

    return actors

def _build_tube_actor_from_sections(sections, radius, color, alpha, res, parcelIDs=None):
    '''
    Takes a list of point arrays (one per morphological section) and builds
    them all into a single vtkPolyData, then applies ONE vtkTubeFilter to it.
    Returns a single vedo Mesh actor.

    Parameters
    ----------
    sections : list
        list containing neuron sections.
    radius : int
        minimum radius for actors to be rendered at.
    color : string
        desired color.
    alpha : float
        desired alpha.
    res : int
        number of triangles to draw around lines to create tube object.
    parcelIDs : list, optional
        parcellation IDs to filter nodes based on. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    points = vtk.vtkPoints()
    lines  = vtk.vtkCellArray()
    pt_offset = 0

    for pts in sections:
        
        #filter points based on desired regions to render axon in if passed
        if parcelIDs:
            keep = np.ones(len(pts), dtype=bool)
            for i, pt in enumerate(pts):
                tenmicron = np.round([x/10 for x in pt]).astype(int).tolist()
                #swap back to ccf axes
                #TODO: keep parcel id info from swc files so i don't have to reindex ccf here for every node its a lot
                allenid = allen_ccf_data[tenmicron[2], tenmicron[1], tenmicron[0]]
                if allenid not in parcelIDs:
                    keep[i] = False
                    continue
            pts = pts[keep]
            
        if len(pts) < 2:
            continue
        
        # create empty polyline, then insert into the cell array to apply the tube filter on
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(pts))
        for i, pt in enumerate(pts):
            points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
            polyline.GetPointIds().SetId(i, pt_offset + i)
        lines.InsertNextCell(polyline)
        pt_offset += len(pts)

    if pt_offset == 0:
        return None

    #add data to polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # apply tube fulter to polydata, set resolution and radius
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polydata)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(res) #vtk tubes render on triangles, more triangles means higher resolution
    tube.CappingOn()
    tube.Update()

    return Mesh(tube.GetOutput(), c=color, alpha=alpha)

#CALLABLES
def create_cell_actors(swcpath, axon='green', dendrite='black', soma='black',
                         skip_dendrite=False, skip_axon=False, neurite_radius=4, soma_radius=15,
                         alpha=1, res=12, regs_to_render=None, mirror_soma=False):
    '''
    create actors for brainrender visualization

    Parameters
    ----------
    swcpath : str
        path to swc file containing reconstructed neuron.
    axon : str, optional
        color to render axon as. The default is 'green'. Color info for axon, dendrite, soma i believe can be found here
        https://vedo.embl.es/site/api/colors/
    dendrite : str, optional
        color to render dendrite as. The default is 'black'.
    soma : str, optional
        color to render soma as. The default is 'black'.
    skip_dendrite : bool, optional
        True to not render dendrite. The default is False.
    skip_axon : bool, optional
        True to not render axon. The default is False.
    neurite_radius : int, optional
        desired radius for neurites. The default is 4.
    soma_radius : int, optional
        desired soma radius. The default is 15.
    alpha : float, optional
        alpha for reconstructions to be rendered at. The default is 1.
    res : int, optional
        desired resolution for brainrender to take a screenshot at. The default is 12.
    regs_to_render : list, optional
        list of allen CCF region abbreviations that you want to render axon in. The default is None/render axon everywhere.
    mirror_soma : bool, optional
        True to mirror soma over midline. The default is False.

    Returns
    -------
    cell_actors : list
        list of Actor objects to be added to brainrender scene.

    '''
    swc_df = pd.read_csv(
        swcpath, comment='#', sep=r'\s+',
        names=['id', 'type', 'x', 'y', 'z', 'r', 'parent', 'parcelID']
    )

    # ── Step 2: axis swap for brainrender display ─────────────────────────────
    swc_df['x'], swc_df['z'] = swc_df['z'].copy(), swc_df['x'].copy()

    #do not include soma if you are rendering axon in specific regions
    include_soma = False if regs_to_render else True
    
    # ── Step 3: build actors from full topology, section-level filter ─────────
    cell_actors = _swc_to_line_actors(
        swc_df,
        axon_color=axon, dendrite_color=dendrite, soma_color=soma,
        skip_dendrite=skip_dendrite, skip_axon=skip_axon, neurite_radius=neurite_radius,
        soma_radius=soma_radius, alpha=alpha, res=res,
        include_soma=include_soma,
        regs_to_render=regs_to_render, mirror_soma=mirror_soma
    )
    return cell_actors

def render_cells(scene, camera='frontal', celldir=[], cellfile=[], celllist=[], screenshot=False, savef=None, scale=None, color_map=None, **kwargs):
    '''
    render cells in brainrender. Pass in a scene with any desired region meshes
    already added, any visualization settings already set, this will just
    draw cells and either render the scene or take a screenshot

    Parameters
    ----------
    celllist : list
        list of cell names to be rendered. default none/no filtering by name
    celldir : str
        Path to directory containing cell swc files, default none if input is a file instead of directory
    cellfile : str
        Path to file of swc to be rendere. default none if rendering files from directory instead of an individual cell
    scene : brainrender Scene
        brainrender Scene object to draw the cells in.
    camera : dict | string
        dictionary containing parameters for brainrender's camera. Or string specifying a built in brainrender camera. default 'frontal'.
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
        keyword arguments to be passed to create_cell_actors

    Returns
    -------
    None.

    '''
    def add_actors(actors, scene):
        for actor in actors:
            scene.add(actor)
    
    #if rendering files from directory, filter based on a list of cell names if passed
    if celldir:
        for cell in tqdm(os.listdir(celldir), desc='rendering cells'):
            
            cellname = cell.split('.')[0]
            fn = os.path.join(celldir, cell)
            
            if celllist:
                if cellname in celllist:
                    kwargs.update({'axon': color_map[cellname]})
                    actors = create_cell_actors(fn, **kwargs)
                    add_actors(actors, scene)
            
            else:
                actors = create_cell_actors(fn, **kwargs)
                add_actors(actors, scene)
        
    #render single file
    if cellfile:
        actors = create_cell_actors(cellfile, **kwargs)
        add_actors(actors, scene)
                
    if screenshot:
        scene.screenshot(savef, camera=camera, scale=scale)
        scene.close()
    
    else:
        scene.render(camera=camera)
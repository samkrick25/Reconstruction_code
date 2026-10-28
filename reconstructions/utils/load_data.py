import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import json
from treelib import Tree
#import brainrender
from brainrender.actors import Neuron
#import vedo
from reconstructions.utils.filedirs import allen_ccf_10um, allen_parcellationpkl, parcellation_mappkl, ptermpkl, bin_ccf
import nibabel as nib
import time
import sys
from collections import defaultdict

MIDLINEZ = 5750
MIDLINEZ_10UM = 570

allen_ccf = nib.load(allen_ccf_10um)
allen_ccf_data = np.asanyarray(allen_ccf.dataobj)
bin_ccf_img = nib.load(bin_ccf)
bin_ccf_data = bin_ccf_img.dataobj
allen_parcellations = pickle.load(open(allen_parcellationpkl, 'rb'))
parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))
allenstr = 'AllenCCF-Annotation-2020-'
pterm = pickle.load(open(ptermpkl, 'rb'))


def load_from_swc(swcdir = None, swc = None):
    '''
    load neurons into a dictionary, read from .swc files

    Parameters
    ----------
    swcdir : TYPE
        DESCRIPTION.

    Returns
    -------
    neurondict: dictionary with keys being cell names and values being dictionaries containing soma, axon, dendrite nodes.

    '''
    neurondict = {}
    if swcdir:
        gen = os.listdir(swcdir)
    if swc:
        gen=[swc]
    for file in tqdm(gen, desc='Loading neurons: '):
        if swcdir:
            fullfile = os.path.join(swcdir, file)
            cellname = file.split('.')[0]
        if swc:
            fullfile = swc
            cellname = file.split('\\')[-1].split('.')[0]
        celldict = {'soma': None, 'axon': [], 'dendrite': []}
        node_to_struct = {}
        with open(fullfile, 'r') as f:
            for line in f: 
                if line.lstrip().startswith('#'):
                    continue
                
                keys = ['sampleNumber', 'structureIdentifier', 'x', 'y', 'z', 'radius', 'parentNumber']
                node_dict = {key: None for key in keys}
                
                toks = line.split(' ')
                
                node_dict['sampleNumber'] = int(toks[0])
                node_dict['structureIdentifier'] = int(toks[1])
                node_dict['x'] = float(toks[2])
                node_dict['y'] = float(toks[3])
                node_dict['z'] = float(toks[4])
                node_dict['radius'] = float(toks[5])
                node_dict['parentNumber'] = int(toks[6])
                
                #track structure id of each node for branch and end points
                node_to_struct[int(toks[0])] = {'structure' : int(toks[1]), 'parent': int(toks[6])}
                
                #check if node is soma, dendrite or axon and add to corresponding key
                match node_dict['structureIdentifier']:
                    #soma
                    case 1:
                        celldict['soma'] = node_dict
                        
                    #axon
                    case 2:
                        celldict['axon'].append(node_dict)
                        
                    #dendrite
                    case 3:
                        celldict['dendrite'].append(node_dict)
                        
                    #5 is branch, 6 is end
                    #these will appear in both axon and dendrite, so need to handle
                    #by referring to the parent node and using its structure id to assign
                    case 5 | 6:
                        
                        #search through parents until you find one that is axon or dendrite
                        #possible edge case is if the node after the soma is a branch, this will skip the node
                        #but i feel like that isn't a thing? hopefully but definitely something to think of a way to fix
                        #ideally you would have to check that node's children but that requires loading the 
                        #whole cell first? idk something to think about at some point hopefully won't be an issue
                        parent = node_dict['parentNumber']
                        parent_compartment = node_to_struct[parent]['structure']
                        while parent_compartment not in (2, 3):
                            parent = node_to_struct[parent]['parent']
                            parent_compartment = node_to_struct[parent]['structure']

                        match parent_compartment:
                            
                            case 2:
                                celldict['axon'].append(node_dict)
                                
                            case 3:
                                celldict['dendrite'].append(node_dict)
                        
            neurondict[cellname] = celldict
            
    return neurondict


def neuronprop(neurondict, mode, cellname, ontlevel='structure'):
    '''
    measure properties of a neuron, either length or endpoint frequency in annotated ccf regions 
    works with neuron dictionary as loaded in by load_neurons, iterate over dictionary and call this on each entry

    Parameters
    ----------
    neurondict : dictionary
        dictionary (rewrite to be OO so i can say smth specific here).
    mode: str
        'length' or 'frequency', desired property to measure about set of reconstructions
    ontlevel: str
        desired ontology level of ccf, default structure
    cellname: name of neuron to be measured
    
    Returns
    -------
    one or two dictionaries containing either endpoint frequency or length in allen CCF-defined regions
    
    '''
    
    if mode == 'length':
        nodes = neurondict[cellname]['axon']
        dendrite = neurondict[cellname]['dendrite']
        somax, somay, somaz = [neurondict[cellname]['soma']['x'], neurondict[cellname]['soma']['y'], 
                               neurondict[cellname]['soma']['z']]
    if mode == 'frequency':
        cell = get_axonal_endpoints(neurondict)
        nodes = cell[cellname]['ends']
        somax, somay, somaz = [cell[cellname]['soma']['x'], cell[cellname]['soma']['y'], 
                           cell[cellname]['soma']['z']]
    
    #just axon nodes
    nodedf = pd.DataFrame(nodes)
    #axon and dendrite nodes, used if an axon node's parent is a dendrite
    #only happens for the first node in an axon, and if in length mode
    #don't care about parent if in frequency mode
    if mode=='length':
        neuritedf = pd.DataFrame(dendrite+nodes)
    
    propdict = defaultdict(np.float64)
    
    #get sign of soma relative to midline
    somaref = MIDLINEZ-somaz
    somahem = np.sign(somaref)
    
    for _, node in nodedf.iterrows():
        
        coords = [node['x'], node['y'], node['z']]

        x, y, z = coords
        
        #get sign of node relative to midline, get laterality of node
        zref = MIDLINEZ-z
        zhem = np.sign(zref)
        
        #index allen ccf volume to get parcellation id if node is in annotated ccf
        region, _ = find_region_annotation(coords, ontlevel)
        if region == 'node not in ccf':
            continue
        else:  
            latregion = 'Ipsilateral '+region if zhem == somahem else 'Contralateral '+region
            
        #calculate euclidean distance from node to its parent, soma will have parentNumber -1
        match mode:
            case 'length':
                #for first node of axon
                if node['parentNumber'] == 1:
                    length = np.sqrt((x-somax)**2+(y-somay)**2+(z-somaz)**2)
                    propdict[latregion] += length
                else:
                    #sometimes axon nodes come from dendrite instead of soma, so need to get access into 
                    #dendrite nodes
                    try:    
                        pnode = neuritedf[neuritedf['sampleNumber'] == node['parentNumber']]
                        px, py, pz = [pnode['x'].values[0], pnode['y'].values[0], pnode['z'].values[0]]
                    except IndexError:
                        print(node, cellname)
                        print()
                    length = np.sqrt((x-px)**2+(y-py)**2+(z-pz)**2)
                    propdict[latregion] = propdict[latregion] + length
                    #technically, if a parent is in a different region than its child this length will count the whole length between the two
                    #as in the child's region, but nodes are usually placed close enough that I don't think this matters

            
            case 'frequency':
                propdict[latregion] += 1
            
    return propdict

def find_region_annotation(coords, ontlevel):                
    '''
    get region annotaion for a node in Allen CCFv3
    
    cerebellum check is an edge case for cells that project within cerebellum but outside of 
    CCF volume since the posterior bound of the CCF doesn't extend to the end of the cerebellum

    Parameters
    ----------
    coords : list
        x,y,z coordinates of the node you want to find region annotation for.
    ontlevel : string
        desired ontology level of the Allen CCF. valid options are: 
        ['organ', 'category', 'division', 'structure', 'substructure']

    Returns
    -------
    str, Allen CCF region abbreviation, or a string saying that the node is in 
    an unannotated part of the ccf/outside ccf

    '''
    tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()
    spc, string = check_if_spinal_cord(coords, bin_ccf_data)
    if spc == True:
        region = 'SpC'
        allenid = 0
        return region, allenid
        
    #posterior cerebellum check
    if string == 'cerebellum':
        region = 'CB'
        allenid = 502
        return region, allenid
        
    if spc == False and string != 'cerebellum':
        try:
            allenid = allen_ccf_data[tenmicron[0], tenmicron[1], tenmicron[2]]
            if allenid == 0:
                return 'node not in ccf', allenid
        except IndexError:
            allenid = 0
            return 'node not in ccf', allenid
        
        #get region abbreviation if node in ccf volume
        else:
            region = get_allen_region(ontlevel, allenid)
            
            return region, allenid

def check_if_spinal_cord(node, binarized_ccf):
    '''
    checks if a node is in the spinal cord

    Parameters
    ----------
    node : list
        x, y, z coordinates of node in ten micron resolution.
    binarized_ccf : array-like
        binarized ccf volume

    Returns
    -------
    bool. True if node is in spinal cord
    string. used for debugging purposes
    '''
    #initially check if node is outside posterior bound of CCF and below cerebellum
    #do this before rounding
    if node[0] > 13192.5 and node[1] > 4642:
        string = 'posterior'
        return True, string
    
    if node[0] > 13192.5 and node[1] < 4642:
        string = 'cerebellum'
        return False, string
    
    #if not, index binarized ccf
    tenmicron = np.round([x/10 for x in node]).astype(int).tolist()
    #below ventral bound of ccf, only nodes that fulfill this condition are those in the spinal cord
    if tenmicron[1] >= 800:
        string = 'below ventral bound'
        return True, string
    
    #in case any spinal cord ones go outside lateral bound of ccf
    if tenmicron[2] >= 1140:
        string = 'outside lateral bound'
        return True, string
    
    try:
        ccf_bool = binarized_ccf[tenmicron[0], tenmicron[1], tenmicron[2]]
    except IndexError:
        print('ya screwed up')

    #in ccf vol
    if ccf_bool == 1:
        string = 'in ccf'
        return False, string
    
    #not in ccf vol, this and the ML check are really only for cells in sample 703070 since the spinal cord stayed attached
    #and cells were traced into spinal cord
    if ccf_bool == 0:
        string = 'not posterior, wrapping'
        return True, string
                    
def get_allen_region(ontlevel, allenid):
    '''
    find allen region associated with a given parcellation id at any ontology level

    Parameters
    ----------
    ontlevel : str
        desired level of the allen ontology to pull.
    allenid : int
        parcellation id to find associated region for

    Returns
    -------
    region : str
        region at desired ontology level associated with a given parcellation id.

    '''
    parcels = parcellation_map[parcellation_map['parcellation_index'] == allenid]
    
    match ontlevel:
        case 'organ':
            region = parcels.loc[parcels['parcellation_term_set_name']=='organ', 'parcellation_term_acronym'].values[0]
        case 'category':
            region = parcels.loc[parcels['parcellation_term_set_name']=='category', 'parcellation_term_acronym'].values[0]
        case 'division':
            region = parcels.loc[parcels['parcellation_term_set_name']=='division', 'parcellation_term_acronym'].values[0]
        case 'structure':
            region = parcels.loc[parcels['parcellation_term_set_name']=='structure', 'parcellation_term_acronym'].values[0]
        case 'substructure':
            region = parcels.loc[parcels['parcellation_term_set_name']=='substructure', 'parcellation_term_acronym'].values[0]
    return region

def get_nodes_in_region(cells, regions, parcellated=True, ontlevel='structure', kind=None, infunc=False, mirror=False):
    '''
    Docstring for get_nodes_in_region
    
    :param cells: Description
    :param regions: Description
    returns a list of nodeIDs that I can then use to pull the specific nodes from coordswapped that have coords for visualizaiton
    or if inputting the non parcellated jsons, will regurn a list of nodes in desired region
    '''
# =============================================================================
#     if infunc:
#         regions=regions[0]
# =============================================================================
        
    match kind:
        case 'bulk':
            nodes = []
            for _, axon in cells.items():
                for node in axon:
                    if parcellated:
                        try: 
                            if node[ontlevel] in regions:
                                nodes.append(node['sampleNumber'])
                        except KeyError:
                            print(node)
                    #if you want not parcellated, then regions has to be the ont id (numbers), if using parcellated can find the region abv
                    if not parcellated:
                        if node['allenId'] in regions:
                            nodes.append(node)
            return nodes
        case 'by_cell':
            cellstonodes = {}
            for cell, axon in tqdm(cells.items(), desc='Finding nodes in region'):
                nodes = []
                for node in axon:
                    if parcellated:
                        try:
                            if node[ontlevel] in regions:
                                nodes.append(node['sampleNumber'])
                        except KeyError:
                            print(node)
                    if not parcellated:
                        if node['allenId'] == None:
                            continue
                        if node['x'] > 13200:
                            continue
                        coords = [node['x'], node['y'], node['z']]
                        tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()
                        try:
                            allenid = allen_ccf_data[tenmicron[0], tenmicron[1], tenmicron[2]]
                        except IndexError:
                            region = 'SpC'
                        else:
                            parcels = parcellation_map[parcellation_map['parcellation_index'] == allenid]
                            region = get_allen_region(ontlevel, parcels)
                        if mirror:
                            if node['z'] < 5700:
                                diff = 5700 - node['z']
                                node['z'] = 5700+diff
                        if region in regions:
                            nodes.append(node)
                cellstonodes[cell] = nodes
            return cellstonodes
    


def load_neurons_from_json(folderpath):
    '''
    load all neurons into a dictionary with cell names as keys and a nested dictionary containing soma, axon, and dendrite
    
    :param folderpath: path containing the reconstruction json files to be loaded
    returns: a dictionary where keys are cell ids and values are a dictionary containin axon, soma, and dendrite info
    '''
    neuron_dict={}
# =============================================================================
#     aidtoreg = {}
#     regtoaid = {}
#     abvtoaid = {}
# =============================================================================
    #somas = {}

    for file in tqdm(os.listdir(folderpath), desc='Loading neurons'):
        filename = os.path.join(folderpath, file)
        with open(filename, 'r') as f:
            fdict = json.load(f)
            #again, since some of the neurons are annotated in different ccf versions, i have to swap some coords around, going to write a coordinate swapper that takes the version and fdict and will swap around coords if needed
            #i think this specific functionality is deprecated now, since I swapped coords and resaved the jsons, going to comment it out
            # ver = fdict['neurons'][0]['annotationSpace']['version']
            # if ver == 2.5:
            #     fdict = coord_swapper(fdict)
            cellname = fdict['neurons'][0]['idString']
            axon = fdict['neurons'][0]['axon']
#            alleninfo = fdict['neurons'][0]['allenInformation']
            soma = fdict['neurons'][0]['soma']
            dendrite = fdict['neurons'][0]['dendrite']
# =============================================================================
#             #build region dictionary
#             for region in alleninfo:
#                 aid = region['allenId']
#                 name = region['name']
#                 acronym = region['acronym']
#                 aidtoreg[aid] = (name, acronym)
#                 regtoaid[name] = aid
#                 abvtoaid[acronym] = aid
# =============================================================================
            neuron_dict[cellname] = {
                'axon': axon,
                'soma': soma,
                'dendrite': dendrite
                }
#            somas[cellname] = soma
            
    return neuron_dict

def get_axonal_endpoints(neurondict):
    '''
    find endpoints of a cells axon and return for analysis of endpoint frequency

    Parameters
    ----------
    neurondict : dictionary
        nested dictionary, first key is the cell name, then keys are 'axon', 'soma', 'dendrite',
        where values correspond to nodes that belong to each structure.

    Returns
    -------
    endsdict : dictionary
        nested dictionary where first key is cell name, then keys are 'ends', 'soma',
        where values correspond to nodes that are ends and soma node.

    '''
    endsdict = {}
    for cell, info in neurondict.items():
        axon = info['axon']
        axonids = [str(node['sampleNumber']) for node in axon]
        soma = info['soma']
        parent_child_dict = {}
        for node in axon:
            nodeID = str(node['sampleNumber'])
            parent = str(node['parentNumber'])
            if isinstance(parent, str) == False:
                continue
            if parent == str(1):
                parent_child_dict[nodeID] = None
            if parent > str(1):
                #some cells have an axon root that is connected to a dendrite node not soma
                #meaning that i can't just set soma parent to None, but have to check if the parent is
                #an axon node, if not then it sets parent to None/sets node as root
                if parent not in axonids:
                    parent_child_dict[nodeID] = None
                else:
                    parent_child_dict[nodeID] = str(parent)
        tree = Tree()
        tree = tree.from_map(parent_child_dict)
        leaves = tree.leaves()
        ends_from_tree = [int(node.identifier) for node in leaves]
        endpoints = [node for node in axon if node['sampleNumber'] in ends_from_tree]
        endsdict[cell] = {'ends': endpoints, 'soma': soma}
    return endsdict
        

def coord_swapper(fdict):
    '''
    swaps x and z coordinates for soma node, axon and dendrite nodes, used to render cells that were annotated in allen ccfv2.5 in ccfv3 space
    
    :param fdict: dictionary, the neuron json that you need coordinates swapped for
    '''
    soma = fdict['neurons'][0]['soma']
    axon = fdict['neurons'][0]['axon']
    dendrite = fdict['neurons'][0]['dendrite']
    somax = soma['x']
    somaz = soma['z']
    soma['x'] = somaz
    soma['z'] = somax
    fdict['neurons'][0]['soma'] = soma
    for node in axon:
        x = node['x']
        z = node['z']
        node['z'] = x
        node['x'] = z
    for node in dendrite:
        x = node['x']
        z = node['z']
        node['z'] = x
        node['x'] = z
    fdict['neurons'][0]['axon'] = axon
    fdict['neurons'][0]['dendrite'] = dendrite
    return fdict

def annotate_swc(swc):
    '''
    annotate swc's with region info for each node, useful for visualizing axon length in specific regions

    Parameters
    ----------
    swc : path
        path to the swc to be annotated.

    Returns
    -------
    pd.DataFrame of swc with column containing region annotation parcellation id.

    '''
    df = pd.read_csv(swc, sep=' ', names=['nodeID','structureIdentifier','x','y','z','radius','parent'],
                     comment='#')
    
    df['parcelID'] = None
    
    nodeparcels = []
    
    for row, vals in df.iterrows():
        coords = [vals['x'], vals['y'], vals['z']]
        
        #ontology level doesn't matter here, i'm annotating with the parcellation index
        #which is at substructure resolution so I'll pass that, but I'm not using any region abbreviations
        try:
            _, allenid = find_region_annotation(coords, 'substructure')
        except TypeError:
            print()
            
        nodeparcels.append(allenid)
        
    df.parcelID = nodeparcels    
        
    return df
    
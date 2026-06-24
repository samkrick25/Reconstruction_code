import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
#from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import json
from treelib import Tree
#import brainrender
from brainrender.actors import Neuron
#import vedo
from reconstructions.utils.filedirs import allen_ccf_10um, allen_parcellationpkl, parcellation_mappkl
import nibabel as nib
import time
import sys
from collections import defaultdict

MIDLINEZ = 5750
MIDLINEZ_10UM = 570

allen_ccf = nib.load(allen_ccf_10um)
allen_ccf_data = np.asanyarray(allen_ccf.dataobj)
allen_parcellations = pickle.load(open(allen_parcellationpkl, 'rb'))
parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))


def get_frequencies_from_dict(neurondict, ontlevel='structure'):
    '''
    right now im only writing this for axon endpoint analysis, ill have to first pull the endpoints thru helper function
    '''
    axonalends = get_axonal_endpoints(neurondict)
    for cell, info in axonalends.items():
        soma = info['soma']
        ends = info['ends']
        freqdict = {}
        somaz = soma['z']
        somaref = MIDLINEZ_10UM - somaz
        #set somahem to - if LH, + if RH, to find lateralization of axonal ends
        somahem = np.sign(somaref)
        for end in ends:
            match ontlevel:
                case 'organ':
                    if end['organ'] is None:
                        continue
                    else:
                        freq_helper(freqdict, end, end['organ'], somahem)
                case 'category':
                    if end['category'] is None:
                        continue
                    else:
                        freq_helper(freqdict, end, end['category'], somahem)
                case 'division':
                    if end['division'] is None:
                        continue
                    else:    
                        freq_helper(freqdict, end, end['division'], somahem)
                case 'structure':
                    try: 
                        if end['structure'] is None:
                            continue               
                        else:
                            freq_helper(freqdict, end, end['structure'], somahem)
                    except KeyError:
                        print(cell, end)
                        continue
                case 'substructure':
                    if end['substructure'] is None:
                        continue
                    else:
                        freq_helper(freqdict, end, end['substructure'], somahem)
                case None:
                    raise ValueError('Please specify desired ontology level for annotation!')
            ser = pd.Series(freqdict, name=cell)
            ser = ser.replace(np.nan, 0)
    return ser

def get_axon_length(neurondict, ontlevel='structure'):
    '''
    get axon length in regions, lateralized, done at one micron resolution, values are in um
    '''
    lengthdict = defaultdict(int)
    axon = neurondict['axon']
    somax, somay, somaz = [neurondict['soma']['x'], neurondict['soma']['y'], neurondict['soma']['z']]
    
    #get sign of soma relative to midline
    somaref = MIDLINEZ-somaz
    somahem = np.sign(somaref)
    
    for node in axon:
        
        coords = [node['x'], node['y'], node['z']]
        x, y, z = coords
        
        #index allen ccf volume to get parcellation id, needs to be 10 um res
        tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()
        allenid = allen_ccf_data[tenmicron[0], tenmicron[1], tenmicron[2]]
        parcels = parcellation_map.loc[allen_parcellations.loc[allenid]['label']]
        region = get_allen_region(ontlevel, parcels)
        
        #get sign of node relative to midline, get laterality of node
        zref = MIDLINEZ-z
        zhem = np.sign(zref)
        latregion = 'Ipsilateral '+region if zhem == somahem else 'Contralateral '+region
        
        #calculate euclidean distance from node to its parent, soma will have parentNumber -1
        if node['parentNumber'] == -1:
            length = np.sqrt((x-somax)**2+(y-somay)**2+(z-somaz)**2)
            lengthdict[latregion] += length
        else:
            pnode = axon[node['parentNumber']]
            px, py, pz = [pnode['x'], pnode['y'], pnode['z']]
            length = np.sqrt((x-px)**2+(y-py)**2+(z-pz)**2)
            lengthdict[latregion] += length
            
    return lengthdict
                    
def get_allen_region(ontlevel, parcels):
    '''
    find region abbreviation from allen parcellation id
    '''
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

def freq_helper(freqdict, end, region, somahem):
    '''
    helper function for my frequency annotator

    Parameters
    ----------
    freqdict : 
        DESCRIPTION.
    end : TYPE
        DESCRIPTION.
    region : TYPE
        DESCRIPTION.
    somahem : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    zend = end['z']
    zref = MIDLINEZ_10UM - zend
    #also, should be - for LH, + for RH
    zhem = np.sign(zref)
    ipsstr = 'Ipsilateral ' + region
    contstr = 'Contralateral ' + region
    if zhem == somahem:
        if ipsstr in freqdict:
            freqdict[ipsstr] += 1
        else:
            freqdict[ipsstr] = 1
    if zhem != somahem:
        if contstr in freqdict:
            freqdict[contstr] += 1
        else:
            freqdict[contstr] = 1
    return
    
def parcellation_annotator(node):
    #round coordinates to 10um resolution
    startround = time.perf_counter()
    coords = [node['x'], node['y'], node['z']]
    coords = [x/10 for x in coords]
    coords = np.round(coords).astype(int).tolist()
    node['x'], node['y'], node['z'] = coords
    endround = time.perf_counter()
    roundtime = endround-startround
    
    #find parcellation index
    startindex=time.perf_counter()
    parcellation_index = allen_ccf_data[node['x'], node['y'], node['z']]
    parcellation_label = parcellation_map.loc[allen_parcellations.loc[parcellation_index]['label']]
    endindex=time.perf_counter()
    indextime = endindex-startindex
    
    
    #annotate each ontology level
    #should figure out why this is so slow and rewrite it to be faster, maybe I can look for the parcellation ID concatenated onto a string?
    startannot = time.perf_counter()
    node['organ'] = parcellation_label.loc[parcellation_label['parcellation_term_set_name']=='organ', 'parcellation_term_acronym'].values[0]
    node['category'] = parcellation_label.loc[parcellation_label['parcellation_term_set_name']=='category', 'parcellation_term_acronym'].values[0]
    node['division'] = parcellation_label.loc[parcellation_label['parcellation_term_set_name']=='division', 'parcellation_term_acronym'].values[0]
    node['structure'] = parcellation_label.loc[parcellation_label['parcellation_term_set_name']=='structure', 'parcellation_term_acronym'].values[0]
    node['substructure'] = parcellation_label.loc[parcellation_label['parcellation_term_set_name']=='substructure', 'parcellation_term_acronym'].values[0]
    endannot = time.perf_counter()
    annottime = endannot-startannot
    
    return roundtime, indextime, annottime
    
def get_node_parcellations(neurondict):
    for cell, info in tqdm(neurondict.items(), desc='Finding parcellations'):
        axon = info['axon']
        dendrite = info['dendrite']
        soma = info['soma']
        rtlist = []
        itlist = []
        atlist = []
        ttlist = []
        for node in axon:
            starttimea = time.perf_counter()
            try:
                rta, ita, ata = parcellation_annotator(node)
            except IndexError:
                axon.remove(node)
            finally:
                rtlist.append(rta)
                itlist.append(ita)
                atlist.append(ata)
            endtimea=time.perf_counter()
            totaltimea=endtimea-starttimea
            ttlist.append(totaltimea)
        for node in dendrite:
            starttimed=time.perf_counter()
            try:
                rtd, itd, atd = parcellation_annotator(node)
            except IndexError:
                dendrite.remove(node)
            else:
                rtd, itd, atd = parcellation_annotator(node)
            finally:
                rtlist.append(rtd)
                itlist.append(itd)
                atlist.append(atd)
            endtimed=time.perf_counter()
            totaltimed=endtimed-starttimed
            ttlist.append(totaltimed)
        parcellation_annotator(soma)
        print(f'Total time to round node coordinates: {np.sum(rtlist)}')
        print(f'Total time to index parcellations: {np.sum(itlist)}')
        print(f'Total time to annotate parcellations: {np.sum(atlist)}')
        print(f'Total time to annotate cell: {np.sum(ttlist)}')
    return

def load_neurons(folderpath):
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

def load_brainrender_neurons(dir, color=None):
    '''
    Docstring for load_brainrender_neurons
    
    :param dir: Description
    :param color: Description
    '''
    neurons = []
    for file in tqdm(os.listdir(dir)):
        filename = os.path.join(dir, file)
        if color is None:
            neuron = Neuron(neuron=filename)
            neurons.append(neuron)
        if color is not None:
            neuron = Neuron(neuron=filename, color=color)
            neurons.append(neuron)
    return neurons

def get_axonal_endpoints(neurondict):
    endsdict = {}
    for cell, info in neurondict.items():
        axon = info['axon']
        soma = info['soma']
        parent_child_dict = {}
        for node in axon:
            nodeID = str(node['sampleNumber'])
            parent = str(node['parentNumber'])
            if parent == str(-1):
                parent_child_dict[nodeID] = None
            if parent > str(0):
                parent_child_dict[nodeID] = str(parent)
        tree = Tree()
        tree = tree.from_map(parent_child_dict)
        leaves = tree.leaves()
        ends_from_tree = [int(node.identifier) for node in leaves]
        endpoints = [node for node in axon if node['sampleNumber'] in ends_from_tree]
        endsdict[cell] = {'ends': endpoints, 'soma': soma}
    return endsdict
        

def get_endpoints_from_file_parcellated(neuronjson):
    '''
    helper func for load_endpoints
    
    :param neuronjson: Description
    '''
    with open(neuronjson, 'r') as f:
        parent_child_dict = {}
        neuron = json.load(f)
        cellname = list(neuron.keys())[0]
        axon = neuron[cellname]['axon']
        #ver = neuron['neurons'][0]['annotationSpace']['version'] #this will be 2.5 if CCFv2.5 is used, 3 if CCFv3
        for node in axon:
            #x = node['x']
            #z = node['z']
            #x and z in ccf2.5 are swapped in ccfv3, so swapping those in any cell annotated in ccfv2.5
            #dont think this is needed anymore, keeping it in in case i get other 2.5 cells?
            # if ver == 2.5:
            #     node['z'] = x
            #     node['x'] = z
            nodeID = str(node['sampleNumber'])
            parent = str(node['parentNumber'])
            if parent == str(-1):
                parent_child_dict[nodeID] = None
            if parent > str(0):
                parent_child_dict[nodeID] = str(parent)
        tree = Tree()
        tree = tree.from_map(parent_child_dict)
        leaves = tree.leaves()
        ends_from_tree = [int(node.identifier) for node in leaves]
        endpoints = [node for node in axon if node['sampleNumber'] in ends_from_tree]
    return endpoints

def get_endpoints_from_file(neuronjson):
    '''
    helper func for load_endpoints
    
    :param neuronjson: Description
    '''
    with open(neuronjson, 'r') as f:
        parent_child_dict = {}
        neuron = json.load(f)
        axon = neuron['neurons'][0]['axon']
        ver = neuron['neurons'][0]['annotationSpace']['version'] #this will be 2.5 if CCFv2.5 is used, 3 if CCFv3
        for node in axon:
            x = node['x']
            z = node['z']
            #x and z in ccf2.5 are swapped in ccfv3, so swapping those in any cell annotated in ccfv2.5
            #dont think this is needed anymore, keeping it in in case i get other 2.5 cells?
            # if ver == 2.5:
            #     node['z'] = x
            #     node['x'] = z
            nodeID = str(node['sampleNumber'])
            parent = str(node['parentNumber'])
            if parent == str(-1):
                parent_child_dict[nodeID] = None
            if parent > str(0):
                parent_child_dict[nodeID] = str(parent)
        tree = Tree()
        tree = tree.from_map(parent_child_dict)
        leaves = tree.leaves()
        ends_from_tree = [int(node.identifier) for node in leaves]
        endpoints = [node for node in axon if node['sampleNumber'] in ends_from_tree]
    return endpoints
# =============================================================================
# 
# def load_endpoints(dir):
#     '''
#     Docstring for load_endpoints
#     
#     :param dir: Description
#     '''
#     allends = []
#     for file in tqdm(os.listdir(dir)):
#         filename = os.path.join(dir, file)
#         endpoints = get_endpoints_from_file(filename)
#         allends.append(endpoints)
#     return allends
# =============================================================================
def swap_swc(fname):
    swc_df = pd.read_csv(
         fname,
         comment='#', 
         sep=r'\s+',
         names=['id', 'type', 'x', 'y', 'z', 'r', 'parent']
    )
    #need to swap x and z coordinates, brainrender swapped their axes so using swcs as we get them will render them rotated 90 deg
    x = swc_df['x']
    z = swc_df['z']
    swc_df['x'] = z
    swc_df['z'] = x
    return swc_df

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
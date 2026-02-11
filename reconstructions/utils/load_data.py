import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import json
import pickle
from treelib import Tree
import brainrender
from brainrender.actors import Neuron
import vedo

MIDLINEZ = 5750

def ver_spec_helper(file_dict, ver=None):
    '''
    helper for my file loader, this does need allen ccf v3 10um loaded in from brainglobeatlas to work correctly
    
    :param file_dict: Description
    :param ver: Description
    '''

    ccf_v3_10um = BrainGlobeAtlas('allen_mouse_10um')
    match ver:
            case 'AA':
                somaz = file_dict['neurons']['soma']['z']
                somaref = MIDLINEZ - somaz
                #sets somahem to - if soma is left hem, + if soma if right hem, will compare with each node down the line
                #to see if a node is ipsi/contra to soma
                somahem = np.sign(somaref)
                somaaid = file_dict['neurons']['soma']['allenId']
                somacomp = ccf_v3_10um.structures[somaaid]['acronym']

                #add cell id to index, create a data frame that is one column, just containing cell soma compartment identified by allenId annotation
                cellId = file_dict['neurons']['idString']

                #build a tree from json, then get endpoints for freq analysis
                parent_child_dict = {}
                nodes = file_dict['neurons']['axon']
                for node in nodes:
                    nodeID = str(node['sampleNumber'])
                    parent = str(node['parentNumber'])
                    if parent == str(-1):
                        parent_child_dict[nodeID] = None
                    if parent > str(0):
                        parent_child_dict[nodeID] = str(parent)
            
            case 'not-AA':
                somaz = file_dict['neurons'][0]['soma']['z']
                somaref = MIDLINEZ - somaz
                #sets somahem to - if soma is left hem, + if soma if right hem, will compare with each node down the line
                #to see if a node is ipsi/contra to soma
                somahem = np.sign(somaref)
                somaaid = file_dict['neurons'][0]['soma']['allenId']
                somacomp = ccf_v3_10um.structures[somaaid]['acronym']
                cellId = file_dict['neurons'][0]['idString']
                #build a tree from json, then get endpoints for freq analysis
                parent_child_dict = {}
                nodes = file_dict['neurons'][0]['axon']
                for node in nodes:
                    nodeID = str(node['sampleNumber'])
                    parent = str(node['parentNumber'])
                    #following is for the first node, the parent for the first node is -1 so edge case control here
                    if parent == str(-1):
                        parent_child_dict[nodeID] = None
                    #then nodes count up from 1, so this gets the parent for each node
                    if parent > str(0):
                        parent_child_dict[nodeID] = str(parent)

    
    tree = Tree()
    tree = tree.from_map(parent_child_dict)
    leaves = tree.leaves()
    ends_from_tree = [int(node.identifier) for node in leaves]
    endpoints = [node for node in nodes if node['sampleNumber'] in ends_from_tree]
    
    return endpoints, somahem, somacomp, cellId


def json_to_freq_from_dir(dir, ver=None):
    '''
    loads in .json files and creates frequency dfs, needs ccfv3 10um loaded in to work correctly
    
    :param dir: directory containing .json files of reconstructions to load

    :param ver: string, default None as it has to be set to 1 of 3 values
                
                'AA' loads in any AA files

                'not-AA' loads all others
                actaully currently reformatting all old AAs so everything will be processed the same

    returns: df with neurons as rows, columns as regions, lateralized, values are raw endpoint counts in each region
    '''

    ccf_v3_10um = BrainGlobeAtlas('allen_mouse_10um')
    data = pd.DataFrame()
    somalocs = []
    for file in tqdm(os.listdir(dir)):
        filename = os.path.join(dir, file)
        with open(filename,'r') as f:
            file_dict = json.load(f)
        
        endpoints, somahem, somacomp, cellId = ver_spec_helper(file_dict, ver)
        somalocs.append((cellId, somacomp))
        freq_dict = {}
        
        for node in endpoints:
            if node['allenId'] is None:
                #some nodes don't have a region annotation, at some point I could recreate this by looking at the coordinates and searching for where it would be?

                continue
            else:
                region = node['allenId']
                z = node['z']
                zref = MIDLINEZ-z
                zsign = np.sign(zref)
                ipsstr = 'Ipsilateral ' + ccf_v3_10um.structures[region]['acronym']
                constr = 'Contralateral ' + ccf_v3_10um.structures[region]['acronym']
                #ipsilateral condition
                if zsign == somahem:
                    if ipsstr in freq_dict:
                        freq_dict[ipsstr] += 1
                    else:
                        freq_dict[ipsstr] = 1
                #contralat condition
                if zsign != somahem:
                    if constr in freq_dict:
                        freq_dict[constr] += 1
                    else:
                        freq_dict[constr] = 1

        match ver:
            case 'AA':
                ser = pd.Series(freq_dict, name=file_dict['neurons']['idString'])
            case 'not-AA':
                ser = pd.Series(freq_dict, name=file_dict['neurons'][0]['idString'])
        #print(ser)
        #this join might be an issue, it seems to be adding duplicate regions so that each neuron has its own column for each region it projects to...
        data = pd.concat([data, ser], join='outer', axis=1)
    data_nonan = data.replace(np.nan, 0)
    datat = data_nonan.T

    return datat, somalocs

def load_neurons(folderpath):
    '''
    Docstring for load_neurons
    
    :param folderpath: Description
    '''
    neuron_dict={}
    aidtoreg = {}
    regtoaid = {}
    abvtoaid = {}
    somas = {}
    freq_df = pd.DataFrame()

    for file in tqdm(os.listdir(folderpath)):
        filename = os.path.join(folderpath, file)
        with open(filename, 'r') as f:
            fdict = json.load(f)
            #again, since some of the neurons are annotated in different ccf versions, i have to swap some coords around, going to write a coordinate swapper that takes the version and fdict and will swap around coords if needed
            ver = fdict['neurons'][0]['annotationSpace']['version']
            if ver == 2.5:
                fdict = coord_swapper(fdict)
            cellname = fdict['neurons'][0]['idString']
            axon = fdict['neurons'][0]['axon']
            alleninfo = fdict['neurons'][0]['allenInformation']
            soma = fdict['neurons'][0]['soma']
            #build region dictionary
            for region in alleninfo:
                aid = region['allenId']
                name = region['name']
                acronym = region['acronym']
                aidtoreg[aid] = (name, acronym)
                regtoaid[name] = aid
                abvtoaid[acronym] = aid
            neuron_dict[cellname] = axon
            somas[cellname] = soma
            
    return neuron_dict, somas, aidtoreg, regtoaid, abvtoaid

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

def get_endpoints(neuronjson):
    '''
    Docstring for get_endpoints
    
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
            if ver == 2.5:
                node['z'] = x
                node['x'] = z
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

def load_endpoints(dir):
    '''
    Docstring for load_endpoints
    
    :param dir: Description
    '''
    allends = []
    for file in tqdm(os.listdir(dir)):
        filename = os.path.join(dir, file)
        endpoints = get_endpoints(filename)
        allends.append(endpoints)
    return allends

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

if __name__ == '__main__':
    #might be able to remove all this and just use this to store loading functions
    #keeping for testing purposes tbh
    

    aa = r'reconstructions\data\json_w_names\AA'
    not_aa = r'reconstructions\data\json_w_names\not-AA'

    aa_raw, aasomalocs = json_to_freq_from_dir(aa, ver='AA')
    not_aa_raw, not_aasomalocs = json_to_freq_from_dir(not_aa, ver='not-AA')

    not_aa_raw['Ipsilateral CUL4 5'] = not_aa_raw['Ipsilateral CUL4, 5']
    not_aa_raw = not_aa_raw.drop(columns=['Ipsilateral CUL4, 5']) 
    
    full_freq = pd.merge(not_aa_raw, aa_raw, how='outer')
    full_freq = full_freq.replace(np.nan, 0)
    full_locs = aasomalocs + not_aasomalocs
    print(full_locs)

    store_full_file = open(r'reconstructions\data\freq_data.pkl', 'ab')
    pickle.dump(full_freq, store_full_file)
    store_full_file.close()
    
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import json
import pickle
from treelib import Tree

MIDLINEZ = 5750

def ver_spec_helper(file_dict, ver=None):
    match ver:
            case 'AA':
                somaz = file_dict['neurons']['soma']['z']
                somaref = MIDLINEZ - somaz
                #sets somahem to - if soma is left hem, + if soma if right hem, will compare with each node down the line
                #to see if a node is ipsi/contra to soma
                somahem = np.sign(somaref)
                somacomp = file_dict['neurons']['soma']['allenId']

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

                #build a tree from json, then get endpoints for freq analysis
                parent_child_dict = {}
                nodes = file_dict['neurons'][0]['axon']
                for node in nodes:
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
    endpoints = [node for node in nodes if node['sampleNumber'] in ends_from_tree]
    
    return endpoints, somahem


def json_to_freq_from_dir(dir, ver=None):
    '''
    loads in .json files and creates frequency dfs
    
    :param dir: directory containing .json files of reconstructions to load
    
    :param ver: string, default None as it has to be set to 1 of 3 values
                
                'AA' loads in any AA files

                'not-AA' loads all others

    returns: df with neurons as rows, columns as regions, lateralized, values are raw endpoint counts in each region
    '''

    data = pd.DataFrame()

    for file in tqdm(os.listdir(dir)):
        filename = os.path.join(dir, file)
        with open(filename,'r') as f:
            file_dict = json.load(f)
        
        endpoints, somahem = ver_spec_helper(file_dict, ver)

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

    return datat

if __name__ == '__main__':
    #might be able to remove all this and just use this to store loading functions
    #keeping for testing purposes tbh
    ccf_v3_10um = BrainGlobeAtlas('allen_mouse_10um')

    aa = r'reconstructions\data\json_w_names\AA'
    not_aa = r'reconstructions\data\json_w_names\not-AA'

    aa_raw = json_to_freq_from_dir(aa, ver='AA')
    not_aa_raw = json_to_freq_from_dir(not_aa, ver='not-AA')

    not_aa_raw['Ipsilateral CUL4 5'] = not_aa_raw['Ipsilateral CUL4, 5']
    not_aa_raw = not_aa_raw.drop(columns=['Ipsilateral CUL4, 5']) 
    
    full_freq = pd.merge(not_aa_raw, aa_raw, how='outer')
    full_freq = full_freq.replace(np.nan, 0)

    store_full_file = open(r'reconstructions\data\freq_data.pkl', 'ab')
    pickle.dump(full_freq, store_full_file)
    store_full_file.close()
    
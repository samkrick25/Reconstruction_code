#notes about json: first key layer is 'comment' and 'neurons'
#for the new reconstructions:
#dict_keys(['id', 'idString', 'DOI', 'sample', 'label', 'annotationSpace', 'annotator', 'proofreader', 'peerReviewer', 'soma', 'axonId', 'axon', 'dendriteId', 'dendrite', 'allenInformation'])
#info for swc/json:
#swc: https://github.com/AllenNeuralDynamics/aind-morphology-utils/blob/67f7e23c9a8d862e62285ca63040c737190cc402/src/aind_morphology_utils/swc.py#L11-L22
#json: https://github.com/AllenNeuralDynamics/aind-morphology-utils/blob/main/src/aind_morphology_utils/writers.py
#this works for recent reconstructions (ones without annotator initial tag in name, wrote specifically for recons gotten from online viewer)


import json
import pandas as pd
import numpy as np
import os
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

ccf_v3_10um = BrainGlobeAtlas('allen_mouse_10um')

dir = r"C:\Users\samkr\OneDrive\Documents\BU\AllenExASPIM\medulla_IRN_PRN_PGRN\json_w_names\new_recons"
data = pd.DataFrame()

for file in os.listdir(dir):
    filename = os.path.join(dir, file)
    with open(filename,'r') as f:
        file_dict = json.load(f)

    #setting z vals for lateralization info
    somaz = file_dict['neurons'][0]['soma']['z']
    #midline est very approximate, no exact way to determine this
    #maybe theres a better way to figure out lateralization but i'm not sure atm
    midlinez = 5750
    somaref = midlinez-somaz
    somasign = np.sign(somaref)

    endpoints = []

    #search for all endpoints (structureIdentifier == 6)
    for node in file_dict['neurons'][0]['axon']:
        if node['structureIdentifier'] == 6:
            endpoints.append(node)

    freq_dict = {}

    for node in endpoints:
        region = node['allenId']
        z = node['z']
        zref = midlinez-z
        zsign = np.sign(zref)
        ipsstr = 'Ipsilateral ' + ccf_v3_10um.structures[region]['acronym']
        constr = 'Contralateral ' + ccf_v3_10um.structures[region]['acronym']
        #ipsilateral condition
        if zsign == somasign:
            if ipsstr in freq_dict:
                freq_dict[ipsstr] += 1
            else:
                freq_dict[ipsstr] = 1
        #contralat condition
        if zsign != somasign:
            if constr in freq_dict:
                freq_dict[constr] += 1
            else:
                freq_dict[constr] = 1
    #print(freq_dict)

    ser = pd.Series(freq_dict, name=file_dict['neurons'][0]['idString'])
    #print(ser)
    #this join might be an issue, it seems to be adding duplicate regions so that each neuron has its own column for each region it projects to...
    data = pd.concat([data, ser], join='inner')
    data_nonan = data.replace(np.nan, 0)
    datat = data_nonan.T

print(len(datat.columns))


# roi = []
# for node in endpoints:
#     region = node['allenId']
#     if region == 773:
#         roi.append(node)
#     else:
#         continue

# for node in roi:
#     print(node['x'], node['y'], node['z'])

#print(file_dict['neurons'][0]['soma']['z'])

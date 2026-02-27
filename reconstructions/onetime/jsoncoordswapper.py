#need to swap x and z coordinates of cells annotated in v2.5 space to visualize correctly
from reconstructions.utils.filedirs import alldir
import json
import os
from tqdm import tqdm

savedir = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\json_coordswapped'

for file in tqdm(os.listdir(alldir)):
    filename = os.path.join(alldir, file)
    with open(filename, 'r') as f:
        neuron = json.load(f)
    ver = neuron['neurons'][0]['annotationSpace']['version']
    if ver == 2.5:
        somax = neuron['neurons'][0]['soma']['x']
        somaz = neuron['neurons'][0]['soma']['z']
        neuron['neurons'][0]['soma']['x'] = somaz
        neuron['neurons'][0]['soma']['z'] = somax
        for node in neuron['neurons'][0]['axon']:
            nodex = node['x']
            nodez = node['z']
            node['x'] = nodez
            node['z'] = nodex
        for node in neuron['neurons'][0]['dendrite']:
            nodex = node['x']
            nodez = node['z']
            node['x'] = nodez
            node['z'] = nodex
    savefile = os.path.join(savedir, file)
    with open(savefile, 'w') as f:
        json.dump(neuron, f)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:40:04 2026
test loading/length measurements
@author: samkr
"""

from reconstructions.utils import load_data as ld
import pandas as pd
from tqdm import tqdm 

swc = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N022-703070.swc"
neurons = ld.load_from_swc(swc=swc)

lengthdict = {}
for neuron, info in tqdm(neurons.items(), desc='finding lengths'):
    lengths = ld.neuronprop({neuron:info}, mode='length', ontlevel='structure', cellname=neuron)
    ser = pd.Series(lengths, name=neuron)
    lengthdict[neuron] = ser
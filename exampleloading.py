# -*- coding: utf-8 -*-
"""
Created on Wed May 6 15:30:58 2026

example script to load in neurons, get projection frequencies, read out targeted regions, and visualize with brainrender

@author: samkr
"""
# %%

#first import necessary packages
import pandas as pd
import numpy as np
#to load and preprocess data
from reconstructions.main import load_data as ld
from reconstructions.main import preprocess_funcs as pf
#to visualize
from brainrender import Scene
from brainrender.actors import Neuron
# %%

#set directory that contains json files of reconstructions (replace with your filepath to jsons)
pathtocelljsons = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\MRN\json"
#set directory for swc files, needed to visualize with brainrender (replace with your filepath to swcs)
pathtocellswc = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\MRN\swc'

# %%
'''
first, lets get the frequency dataframe for your neurons
to do this, I have a function that will index the allen ccf volume for each node, then annotate that node with its region at
all levels of the ccf ontology. i find structure is the most informative and specific but you can look at any level if you want.
this step may take a while to run depending on how many neurons you have, takes about 45 minutes to run on my dataset of n=103
'''

#to do this, you first need to load the neurons into a dictionary where the keys are the neuron ID, and values are the full json
jsondict = ld.load_neurons(pathtocelljsons)
# %%
#then, this function will look at each node, index into the ccf volume, and annotate each node with its region info at all levels of ccf ontology
#this function can take a while to run with a large number of cells, it is recommended to do these first two steps in a separate script
#then pickle the resulting dictionary and save to be reopened in a different script when you want to look at frequency information 
#this function modifies your existing dictionary in place, no need to save to a different variable
ld.get_node_parcellations(jsondict)

# %%
#now, use the following code to get a dataframe containing the number of endpoints in each targeted region for each cell
frequencydf = pd.DataFrame()
for cell in jsondict:
    freqseries = ld.get_frequencies_from_dict({cell: jsondict[cell]}, ontlevel='structure') #you can change ontlevel to your desired ontology level
    frequencydf = pd.concat([frequencydf, freqseries], join='outer', axis=1)
#replace NaNs (regions that aren't targeted by the given cell) with 0s
frequencynonan = frequencydf.replace(np.nan, 0).T
# %%

#then, you can use this function to output a series of targeted regions for a neuron
n097 = pf.get_targeted_regions(frequencynonan, 'N097-709222-DS')
print(n097)

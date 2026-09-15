# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:41:04 2026
pheno/geno analyses
@author: samkr
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import os

savedir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots'
nsf = 'normgp.png'
sf = 'nonormgp.png'

genos = ['AiP1999/AiE2255', 'AiP1996/AiE2199', 'AiP1836/AiE2256', 'Calb2', 'Cart', 'Crh', 'Dbh', 'Gal', 'Grp', 'Ntrk1', 'Sim1',
         'Som', 'Vgat', 'Vglut1', 'Vglut2', 'Retroorbital AAV2/1', 'WT RV (?)', 'ALM antero AAV']

mossy = [5,5,3,0,0,2,1,0,0,2,1,1,0,0,0,0,0,3]
premotor = [5,0,2,0,0,0,0,0,0,0,2,1,2,0,1,1,1,7]
sensory = [1,0,1,0,3,2,0,0,0,0,0,2,6,0,3,2,0,4]
sensorimotor = [0,0,1,0,0,0,0,0,0,0,0,2,4,0,0,0,0,1]
GRN = [2,1,2,1,0,0,0,4,5,0,0,0,9,1,3,1,1,5]
forebrain = [0,0,0,0,0,0,3,0,0,0,0,0,1,0,0,0,0,0]
vestibular = [0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1]

phenos = np.array([mossy, premotor, sensory, sensorimotor, GRN, forebrain, vestibular]).T
phenoLabels = ['mossy', 'premotor', 'sensory', 'sensorimotor', 'GRN', 'forebrain', 'vestibular']

GP = pd.DataFrame(data=phenos, index=genos, columns=phenoLabels)
Pnorm = GP.div(GP.sum(axis=1), axis=0)
GPT = GP.T
Gnorm = GPT.div(GPT.sum(axis=1), axis=0)

pColors = {'sensory': 'firebrick', 'premotor': 'royalblue', 'mossy': 'seagreen', 'GRN': 'purple', 'sensorimotor': 'pink',
           'vestibular': 'navy', 'forebrain': 'orange'}
pmap = ListedColormap(list(pColors.values()))


gColors = {'AiP1999/AiE2255': 'red', 'AiP1996/AiE2199': 'pink', 'AiP1836/AiE2256': 'maroon', 'Calb2': 'cyan',
           'Cart':'blue', 'Crh': 'navy', 'Dbh': 'orange', 'Gal': 'seagreen', 'Grp': 'olive', 'Ntrk1': 'lightgreen',
           'Sim1': 'yellow', 'Som': 'orchid', 'Vgat': 'sienna', 'Vglut1': 'mediumpurple', 'Vglut2': 'purple',
           'Retroorbital AAV2/1': 'magenta', 'WT RV (?)': 'slategrey', 'ALM antero AAV': 'cornflowerblue'}
gmap = ListedColormap(list(gColors.values()))

pDict = GP.to_dict(orient='list')
gDict = GP.T.to_dict(orient='list')

#stacked, phenos in genos, normalized
fign, (gaxn, paxn) = plt.subplots(1,2, figsize=(20,8), layout='constrained')
Pnorm.plot(kind='bar', stacked=True, ax=gaxn, colormap=pmap, legend=False)
#gaxn.legend(loc='upper left')
gaxn.tick_params(rotation=90, axis='x')

pbot = np.zeros(7)
Gnorm.plot(kind='bar', stacked=True, ax=paxn, colormap=gmap, legend=False)
#paxn.legend(loc='upper right')
paxn.tick_params(rotation=90, axis='x')

fign.supylabel('# cells')
gaxn.set_title('Phenotype distribution across genotypes')
paxn.set_title('Genotype distribution across phenotypes')

fign.savefig(os.path.join(savedir,nsf))

#stacked, not normalized 
fig, (gax, pax) = plt.subplots(1,2, figsize=(20,8), layout='constrained')
GP.plot(kind='bar', stacked=True, ax=gax, colormap=pmap)
gax.legend(loc='upper left')
gax.tick_params(rotation=90, axis='x')

pbot = np.zeros(7)
GP.T.plot(kind='bar', stacked=True, ax=pax, colormap=gmap)
pax.legend(loc='upper right')
pax.tick_params(rotation=90, axis='x')

fig.supylabel('# cells')
gax.set_title('Phenotype distribution across genotypes')
pax.set_title('Genotype distribution across phenotypes')

fig.savefig(os.path.join(savedir,sf))
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 03:46:58 2026

@author: economolab
"""

from brainrender import Scene
from brainrender.actors import Neuron
from reconstructions.utils import cameras as cams
import json
from itertools import cycle, islice
import os
from tqdm import tqdm

GRNcelllist = ['AA0503', 'AA1329', 'AA1331', 'AA1460', 'AA1513', 'AA1518', 'AA1521', 'AA1535', 'AA1600', 'AA1602', 'AL0006', 'AL0007', 'AL0008', 'AL0019', 'N004-674185-DS', 'N005-674185', 'N006-703070', 'N008-653980', 'N010-703070', 'N011-703070', 'N014-703070', 'N016-703070', 'N023-703070', 'N023-706301-KA', 'N029-703070', 'N030-703070', 'N031-674185-FMR', 'N032-674185-IB', 'N034-674191-IB', 'N036-674191-SP', 'N039-674185-IB', 'N042-674185-SP', 'N046-658735-AR', 'N051-715345-SA', 'N052-653159-HD', 'N053-653159-YP', 'N054-715345-AP', 'N058-653159-PC', 'N059-653159-AK', 'N059-665081-KA', 'N061-665081', 'N063-665081-AS', 'N064-665081-JN', 'N066-665081-YP', 'N112-708369-JN', 'N126-708369-HD', 'N132-708369-SA']
ccf_scenesag = Scene(atlas_name='allen_mouse_10um', root=False)
XIIcelllist = ['AA0503', 'AA1196', 'AA1405', 'AA1460', 'AA1521', 'AA1535', 'N003-703070', 'N004-674185-DS', 'N005-674185', 'N015-703070', 'N019-703070', 'N021-703070', 'N024-703070', 'N032-674185-IB', 'N034-674191-IB', 'N036-674191-SP', 'N039-674185-IB', 'N054-715345-AP', 'N058-653159-PC', 'N059-653159-AK', 'N063-665081-AS', 'N064-665081-JN', 'N108-708369-VM', 'N130-708369-YP']
# =============================================================================
# root = ccf_scenesag.get_actors()[0]
# root._needs_silhouette = False
# =============================================================================
cellcolors = ['blue','red','orange','green','purple','cyan','magenta']
output = list(islice(cycle(cellcolors), len(GRNcelllist)))
#ccf_scenesag.add_brain_region('MRN', color='pink', alpha=0.3, silhouette=False)
ccf_scenesag.add_brain_region('XII', color='lightpink', alpha=0.3, silhouette=False)
ccf_scenesag.add_brain_region('IRN', color='red',alpha=0.15,silhouette=False)
#ccf_scenesag.add_brain_region('GRN', color='green', alpha=0.15,silhouette=False)
ccf_scenesag.add_brain_region('PARN', color='blue',alpha=0.15,silhouette=False)

direct = r"C:\Data\reconstructions\medulla_IRN_PRN_PGRN\medulla_IRN_PRN_PGRN\aaswc" #r"C:\Data\reconstructions\medulla_IRN_PRN_PGRN\medulla_IRN_PRN_PGRN\swccoordswapped"
neuronlist = []
pngsavedir = r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\images\IRNPARNxiicells\top"
for file, color in tqdm(zip(os.listdir(direct),output),desc='loading Neurons'):
    filestring = file.split('.')[0]
    cellname = filestring#[3:]
    if cellname in XIIcelllist:    
        filename = os.path.join(direct, file)
        print('adding neuron')
        neuron=Neuron(neuron = filename ,color=color)
        ccf_scenesag.add(neuron)
    
        print('neuron added')
        topfile = os.path.join(pngsavedir, filestring+'top.png')
      
        print(f'horizontal screenshot taken for {cellname}')
        sagfile = os.path.join(pngsavedir, filestring+'sag.png')
        ccf_scenesag.screenshot(name=topfile, scale=5, camera=cams.MYtopcam)
        ccf_scenesag.remove(neuron)
        print(f'saggital screenshot taken for {cellname}')
#ccf_scene.render(camera=cams.sagcam)
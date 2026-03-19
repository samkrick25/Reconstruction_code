
import nibabel as nib
import numpy as np
from reconstructions.utils.filedirs import allen_ccf_10um, allen_parcellation, allen_2017_to_2020, allcoordswapped
import pandas as pd
from reconstructions.utils import load_data as ld

allen_ccf = nib.load(allen_ccf_10um)

allen_ccf_data = np.asanyarray(allen_ccf.dataobj)

#print(allen_ccf_data.dtype)
#allen_ccf_data = allen_ccf.get_fdata()

# =============================================================================
# allen_parcellations = pd.read_csv(allen_parcellation, index_col='parcellation_index')
# 
# allen_parcel_map = pd.read_csv(allen_2017_to_2020, index_col = 'parcellation_label')
# root = allen_parcel_map.loc[allen_parcellations.loc[987]['label']]
# organ_label = root.loc[root['parcellation_term_set_name']=='organ', 'parcellation_term_acronym']
# print(organ_label.iloc[0])
# 
# =============================================================================


neurondict = ld.load_neurons(allcoordswapped)
max_xcoords = []
# =============================================================================
# for cell, info in neurondict.items():
#     axon = info['axon']
#     x_coords = [(node['x'], node['allenId']) for node in axon]
#     
#     max_xcoords.append(max(x_coords, key=lambda item: item[0]))
# =============================================================================
ld.get_node_parcellations(neurondict)

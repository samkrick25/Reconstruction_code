from reconstructions.utils.filedirs import parcellation_mappkl, allen_parcellationpkl
import pickle
import pandas as pd

#right now this will write a dictionary that contains each unique structure as keys
#and other ontological information as values (parcellation label, division/category/organ)
#at some point should think of a way to get ontology info from structure name, allensdk?
parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))
allen_parcellations = pickle.load(open(allen_parcellationpkl, 'rb'))

unique_parcellations = []
for label in parcellation_map.index:
    labelid = label.split('-')[-1]
    if labelid not in unique_parcellations:
        unique_parcellations.append(labelid)
    else:
        continue
    
ccf_to_parcel = pd.DataFrame(data=allen_parcellations.index.values, index=allen_parcellations['label'].values, columns=['parcellation_index'])


structure_to_ont = {}
for label in unique_parcellations:
    annotstr = 'AllenCCF-Annotation-2020-'
    labelstr = annotstr + label
    labeldf = parcellation_map.loc[labelstr]
    #this will probably throw an error, will have to find a way to handle regions that will get an annotation
    #but don't have structure level ontology (those that are marked like P-unassigned)
    structure = labeldf.loc[labeldf['parcellation_term_set_name']=='structure', 'parcellation_term_acronym'].values[0]
    
    ontdict = {}
    acrodict = {}
    namedict = {}
    
    acrodict['organ'] = labeldf.loc[labeldf['parcellation_term_set_name']=='organ', 'parcellation_term_acronym'].values[0]
    acrodict['division'] = labeldf.loc[labeldf['parcellation_term_set_name']=='division', 'parcellation_term_acronym'].values[0]
    acrodict['substructure'] = labeldf.loc[labeldf['parcellation_term_set_name']=='substructure', 'parcellation_term_acronym'].values[0]
    acrodict['category'] = labeldf.loc[labeldf['parcellation_term_set_name']=='category', 'parcellation_term_acronym'].values[0]
    
    namedict['organ'] = labeldf.loc[labeldf['parcellation_term_set_name']=='organ', 'parcellation_term_name'].values[0]
    namedict['division'] = labeldf.loc[labeldf['parcellation_term_set_name']=='division', 'parcellation_term_name'].values[0]
    namedict['substructure'] = labeldf.loc[labeldf['parcellation_term_set_name']=='substructure', 'parcellation_term_name'].values[0]
    namedict['category'] = labeldf.loc[labeldf['parcellation_term_set_name']=='category', 'parcellation_term_name'].values[0]
    
    ontdict['parcelid'] = labeldf.index[0]
    ontdict['acronymInfo'] = acrodict
    ontdict['nameInfo'] = namedict
    ontdict['name'] = labeldf.loc[labeldf['parcellation_term_set_name']=='structure', 'parcellation_term_name'].values[0]
    
    structure_to_ont[structure] = ontdict

savefile = r'reconstructions\data\structure_ont_info.pkl'
pickle.dump(structure_to_ont, open(savefile, 'wb'))

# =============================================================================
# from reconstructions.utils import load_data
# from reconstructions.utils.filedirs import allcoordswapped
# import json
# import os
# 
# =============================================================================
# =============================================================================
# _, _, aidtoreg, _, _ = load_data.load_neurons(allcoordswapped)
# 
# filename = 'aidtoreg.json'
# savefolder = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data'
# savepath = os.path.join(savefolder, filename)
# try:
#     with open(savepath, 'w') as f:
#         json.dump(aidtoreg, f)
# except IOError as e:
#     print(f'Error saving file {e}')
# =============================================================================

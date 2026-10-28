# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:27:22 2026
neurom morphometric feature analysis, pull morphometric features and 
create large feature matrix w/everything
combine this with both length and endpoints in region i think, just throw
as much information as i can at a thing and see what happens
@author: samkr
"""

#%% imports
from reconstructions.utils.filedirs import lengthspkl, frequenciespkl, M2
from reconstructions.utils.cellLists import to_merge_L, cells_by_pheno, to_merge_F
from reconstructions.utils.preprocess_funcs import merge_regions
import pickle
import pandas as pd
import numpy as np
import re
import json

#%% function defs
def drop_regex(df, regex):
    to_drop = np.ones(len(df.columns), dtype=bool)
    for i, reg in enumerate(df.columns):
        match = re.findall(regex, reg)
        if match:
            to_drop[i] = 0 
    return df.loc[:, to_drop]

def group_regions(df, to_merge):
    grouped = df.copy()
    for group, regs in to_merge.items():
        groupsums = df[regs].sum(axis=1)
        grouped[group] = groupsums
        grouped.drop(regs, axis=1, inplace=True)
    return grouped

#create a mapping from cell name to pheno identity
def map_val_to_key(d):
    nd = {}
    for k, v in d.items():
        for val in v:
            nd[val]=k
    return nd

def morpho_to_df(morphfile):
    '''
    specific loading for neurom stats json into dataframe, writing each feature individually to avoid MultiIndex

    Parameters
    ----------
    morphfile : str
        path to neurom stats output as a json.

    Returns
    -------
    M : pd.DataFrame
        neurom stats output, cells as rows, features as columns.

    '''
    with open(morphfile, 'r') as f:
        M_pre = json.load(f)
        morph = {}

        #remove .swc to integrate with L and F
        for key, value in M_pre.items():
            key = key.split('.')[0]
            morph[key] = value
            
        #rewrite dictionary to make it easy to convert into dataframe
        M = {}

        for cellname, aggregate in morph.items():
            newdict = {}
            for kind, features in aggregate.items():
                for feature, value in features.items():
                    key = kind+'_'+feature
                    newdict[key] = value
            M[cellname] = newdict
            
        M = pd.DataFrame(M).T
        
        return M

#%% loading and preprocessing of length and frequency data
L = pickle.load(open(lengthspkl,'rb')).T
F = pickle.load(open(frequenciespkl, 'rb')).T

#merge regions on lateralization, maybe shouldn't do this and keep laterality info?
L = merge_regions(L)
F = merge_regions(F)

#%%
#drop any regions that are 0s in all cells (shouldn't be the case)
L = L.loc[:, (L != 0).any(axis=0)]
F = F.loc[:, (F != 0).any(axis=0)]

#drop unassigned stuff, not meaningful for this purpose
string = 'unassigned'
L = drop_regex(L, string)
F = drop_regex(F, string)

#create region groupings
L = group_regions(L, to_merge_L)
F = group_regions(F, to_merge_F)

#%% rename columns of L and F to have length and ends respectively

L = L.rename(lambda x: x+' '+'length', axis=1)
F = F.rename(lambda x: x+' '+'ends', axis=1)

#%% join length and freq into projection data matrix

P = L.join(F, how='outer')

#%% read in neurom morphometrics

M = morpho_to_df(M2)

#%% merge morphometric and projection based features into one matrix

F = M.join(P, how='outer')

#%% remove duplicate cell

#i should rerun the length calculation to not have this here in the first place
#maybe do that overnight or smth
F.drop('N044-674191', axis=0, inplace=True)

#%% write phenotype column

phenos = []
cells_to_pheno = map_val_to_key(cells_by_pheno)
for cell, vals in F.iterrows():
    pheno = cells_to_pheno[cell]
    phenos.append(pheno)
    
F['phenotype label'] = phenos

#%% drop features that have low importance from random forest classification
todrop1 = ['axon_max_partition_asymmetry',
       'basal_dendrite_std_section_strahler_orders',
       'basal_dendrite_std_section_tortuosity', 'axon_sum_number_of_sections',
       'basal_dendrite_max_section_term_branch_orders',
       'axon_std_section_strahler_orders',
       'basal_dendrite_std_remote_bifurcation_angles',
       'morphology_total_aspect_ratio__projection_plane:yz',
       'basal_dendrite_sum_number_of_sections',
       'basal_dendrite_mean_section_term_lengths',
       'basal_dendrite_sum_number_of_forking_points',
       'basal_dendrite_std_partition_asymmetry',
       'axon_std_section_term_lengths',
       'basal_dendrite_max_partition_asymmetry',
       'basal_dendrite_min_section_tortuosity',
       'all_max_segment_meander_angles', 'all_max_bifurcation_partitions',
       'basal_dendrite_max_remote_bifurcation_angles',
       'basal_dendrite_sum_principal_direction_extents',
       'axon_max_segment_meander_angles',
       'basal_dendrite_max_bifurcation_partitions',
       'basal_dendrite_sum_number_of_leaves', 'ventricular ends', 'ventricular length',
       'all_min_section_tortuosity', 'axon_min_section_tortuosity', 'all_max_section_tortuosity', 'all_std_section_branch_orders',
              'all_max_section_term_branch_orders',
              'morphology_min_trunk_section_lengths', 'all_sum_number_of_leaves',
              'all_total_total_area', 'axon_mean_section_bif_lengths',
              'axon_std_section_tortuosity',
              'basal_dendrite_max_partition_asymmetry_length',
              'axon_std_partition_asymmetry',
              'basal_dendrite_mean_section_bif_lengths', 'all_total_total_volume',
              'basal_dendrite_std_section_lengths',
              'axon_max_section_term_branch_orders', 'axon_std_section_branch_orders',
              'axon_min_remote_bifurcation_angles', 'MY/P sensory ends', 'MY/P sensory length',
              'all_mean_section_bif_branch_orders',
              'all_mean_partition_asymmetry_length',
              'morphology_std_trunk_section_lengths',
              'axon_min_partition_asymmetry_length',
              'basal_dendrite_total_total_area',
              'basal_dendrite_min_remote_bifurcation_angles',
              'basal_dendrite_mean_partition_asymmetry_length',
              'axon_sum_number_of_leaves',
              'basal_dendrite_min_partition_asymmetry_length',
              'basal_dendrite_mean_section_branch_orders',
              'axon_std_segment_meander_angles', 'all_mean_section_tortuosity',
              'basal_dendrite_mean_section_lengths',
              'all_sum_number_of_forking_points',
              'basal_dendrite_std_section_term_branch_orders',
              'axon_sum_number_of_bifurcations',
              'basal_dendrite_max_segment_meander_angles',
              'axon_mean_section_term_branch_orders', 'all_sum_number_of_sections',
              'all_mean_section_bif_lengths', 'axon_mean_section_branch_orders',
              'all_mean_section_term_lengths',
              'basal_dendrite_mean_bifurcation_partitions',
              'basal_dendrite_std_section_branch_orders',
              'morphology_total_aspect_ratio__projection_plane:xz',
              'basal_dendrite_mean_section_term_branch_orders',
              'axon_max_remote_bifurcation_angles', 'axon_total_total_area',
              'basal_dendrite_mean_remote_bifurcation_angles',
              'all_mean_remote_bifurcation_angles',
              'basal_dendrite_mean_segment_meander_angles',
              'axon_min_segment_meander_angles', 'all_std_section_tortuosity',
              'basal_dendrite_max_section_tortuosity', 'all_max_partition_asymmetry',
              'axon_mean_section_term_lengths',
              'basal_dendrite_mean_partition_asymmetry',
              'axon_sum_number_of_forking_points', 'all_sum_number_of_bifurcations',
              'axon_mean_section_bif_branch_orders',
              'all_mean_section_strahler_orders',
              'basal_dendrite_sum_number_of_bifurcations',
              'all_max_remote_bifurcation_angles', 'axon_max_bifurcation_partitions',
              'all_mean_section_lengths', 'basal_dendrite_min_segment_meander_angles',
              'basal_dendrite_std_bifurcation_partitions',
              'basal_dendrite_mean_section_bif_branch_orders',
              'morphology_mean_trunk_section_lengths',
              'basal_dendrite_total_total_length', 'all_std_section_term_lengths',
              'basal_dendrite_std_segment_meander_angles',
              'basal_dendrite_total_total_volume',
              'all_min_partition_asymmetry_length',
              'axon_mean_partition_asymmetry_length', 'all_std_section_lengths',
              'all_std_segment_meander_angles',
              'basal_dendrite_mean_section_strahler_orders',
              'axon_mean_section_strahler_orders', 'all_min_segment_meander_angles',
              'all_std_section_strahler_orders', 'all_mean_section_branch_orders',
              'morphology_mean_trunk_angles', 'morphology_max_trunk_angles',
                     'morphology_min_trunk_angles', 'morphology_std_trunk_angles'
       ]

F.drop(todrop1, axis=1, inplace=True)

#%% choose features based on gini importance, drop correlated features or features that are measuring the same thing

todrop = ['basal_dendrite_total_total_volume','basal_dendrite_total_total_length', 'basal_dendrite_std_remote_bifurcation_angles',
          'basal_dendrite_sum_principal_direction_extents','basal_dendrite_std_section_term_branch_orders', 'basal_dendrite_mean_segment_meander_angles',
          'basal_dendrite_mean_partition_asymmetry','basal_dendrite_mean_bifurcation_partitions','axon_total_total_length',
          'axon_std_section_term_lengths','axon_mean_partition_asymmetry','all_total_volume_density', 'all_total_total_volume',
          'all_sum_principal_direction_extents','all_std_section_term_lengths','all_std_section_term_branch_orders','all_std_remote_bifurcation_angles',
          'axon_mean_segment_meander_angles','all_mean_section_tortuosity','all_mean_partition_asymmetry','all_mean_bifurcation_partitions',
          'all_max_partition_asymmetry_length']

F.drop(todrop, axis=1, inplace=True)
#%% pickle and save

sd = r'reconstructions\data\IRNPARN_cells\features_v3.pkl'

pickle.dump(F, open(sd, 'wb'))
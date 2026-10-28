# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:21:15 2026
supervised classification
@author: samkr
"""

from reconstructions.utils.filedirs import F3, allcoordswapped
from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from reconstructions.utils.preprocess_funcs import rem_zero_var, map_val_to_key
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image, display
from collections import defaultdict
from reconstructions.utils.cellLists import pColors
#%%
#load feature matrix
F = pickle.load(open(F3, 'rb'))

#these features resulted in some nans and im not sure how to interpret 
#trunk angle with the values that are given, dropping for now i might reinclude later
nacols = ['morphology_mean_trunk_angles', 'morphology_max_trunk_angles',
       'morphology_min_trunk_angles', 'morphology_std_trunk_angles']

#F = F.drop(nacols, axis=1)

#%% iteratively dropping things with low importance

#no need to run this cell

# < 1e-3 mean gini impurity loss
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
              'all_std_section_strahler_orders', 'all_mean_section_branch_orders'
       ]

F = F.drop(todrop1, axis=1)
#%% feature groupings
#i think here it might be helpful to group features by their correlations, for importance visualization later? idk im just throwing a bunch
#of stuff at a classifier, some stuff will be more important than others

#%%

Flabel = F['phenotype label']

pheno_map = {'sensory':0, 'premotor':1, 'sensorimotor':2, 'GRN':3, 'mossy':4, 'forebrain':5, 'vestibular':6}

Flabel.replace(pheno_map, inplace=True)

Ffeat = F.drop('phenotype label', axis=1)

groups_exaspim = [cell.split('-')[1] for cell, row in Ffeat.iterrows() if cell[0]!='A']

groups_mouselight = ['171031', '180130', '180801', '180801', '180801', '190527', '190527', 
                     '190527', '190906', '190906', '190906', '190906', '200415', '201126', '200915', 
                     '200915', '210317', '210317', '210317', '210317', '210317', '220530', '220530',
                     '220530', '220530']

groups_sample = groups_mouselight + groups_exaspim


#some of the features I pulled are just the same across all cells, drop them here
Ffeat = rem_zero_var(Ffeat)

#stratified traintestsplit, control for different samples
#maybe should also do this for different tracers, and do it normally and see what effects
#diff samples or tracers actually has, but i don't have tracer info for the MouseLight cells

#no sample/tracer controlling
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
for _ in range(100):
    Xtr, Xte, Ytr, Yte = train_test_split(Ffeat, Flabel, stratify=Flabel)
    Xtrain.append(Xtr)
    Ytrain.append(Ytr)
    Xtest.append(Xte)
    Ytest.append(Yte)
    

sgkf = StratifiedGroupKFold(n_splits=10)

#split based on sample
sgkf_sample = sgkf.split(Ffeat, Flabel, groups=groups_sample)

#%% count how many times samples show up in training/testing set

def count_entries(l):
    counts = defaultdict(int)
    for df in l:
        for r, v in df.iterrows():
            counts[r]+=1
            
    return counts

test = count_entries(Xtest)
train = count_entries(Xtrain)
#%% hyperparameter tuning
pdist = {'n_estimators': randint(100,1000),
         'max_depth': randint(3,50),
         'min_samples_split': randint(2,10),
         'min_samples_leaf': randint(1,5),
         'max_samples': randint(90,111),
         }

#make random forest classifier
clf =  RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1)

psearch = RandomizedSearchCV(clf, pdist, n_iter=50, n_jobs=-1, random_state=42, scoring='accuracy', refit='accuracy')

psearch = psearch.fit(Ffeat, Flabel)

best_clf = psearch.best_estimator_

print('Best hyperparams:', psearch.best_params_)
#output {'max_depth': 19, 'max_samples': 93, 'min_samples_leaf': 2, 'min_samples_split': 9, 'n_estimators': 487}
#%% classify with best params

rf = RandomForestClassifier(max_depth=19, max_samples=92, min_samples_leaf=2, min_samples_split=9, n_estimators=487, class_weight='balanced_subsample')
scores=[]
wrong=[]
classifications = defaultdict(list)
for i in range(100):
    rf.fit(Xtrain[i], Ytrain[i])
    #pred = pd.DataFrame(data=clf.predict(Xtest[i]), index=Xtest[i].index)
    ypred = rf.predict(Xtest[i])
    acc = ypred-Ytest[i]
    w = []
    for e, a in enumerate(acc):
        if a != 0 :
            w.append(e)
            classifications[Xtest[i].iloc[e].name].append({'phenotype': Ytest[i].iloc[e], 'prediction': ypred[e], 'iteration':i+1})
    wrong.append(Ytest[i].iloc[w].index)
    score = rf.score(Xtest[i], Ytest[i])
    scores.append(score)

plt.plot(scores)

#%% look at how many times classifier is getting cells wrong and which cells they are
#this is correct, since I split the data and am testing on 
cells_wrong = defaultdict(int)
for l in wrong:
    for cell in l:
        cells_wrong[cell] += 1

#then divide these values by how many times each thing shows up in the testing set
wrong_pct = defaultdict(int)
for k, v in cells_wrong.items():
    wrong_pct[k] = v/test[k]
wrong_pct_sort = dict(sorted(wrong_pct.items(), key=lambda x:x[1], reverse=True))

#filter for cells that are labeled wrong >10% of the time they appear in testing set
wrong_pct2 = defaultdict(int)
for k, v in wrong_pct.items():
    if v>0.5:
        wrong_pct2[k]=v

fig_pct, ax_pct = plt.subplots(figsize=(12, 6))
ax_pct.bar(wrong_pct_sort.keys(), wrong_pct_sort.values())
ax_pct.set_ylabel('% times classified wrong')
plt.tick_params(axis='x', rotation=90)

#%% plot misclassifications of wrong cells

from matplotlib.patches import Rectangle
import os

label_to_pheno = map_val_to_key(pheno_map)

thresh_pct_wrong = 0.2

for cell, scores in classifications.items():
    if wrong_pct[cell] > 0.2:     
        label = scores[0]['phenotype']
        predictions = [score['prediction'] for score in scores]
        xlabels = np.arange(0,7,1)
        ylabels = np.zeros(len(xlabels))
        labels, counts = np.unique(predictions, return_counts=True)
        for i, l in enumerate(labels):
            try:
                ylabels[l] = counts[i]
            except IndexError:
                print()
        fig, ax = plt.subplots()
        ax.bar(xlabels, ylabels, color='k')
        target_index = label
        rect = Rectangle((target_index-0.4, 0), width=0.8, height=test[cell], color=pColors[label_to_pheno[label]],
                         alpha=0.3, zorder=1)
        ax.add_patch(rect)
        fig.suptitle(cell)
        fig.show()
        sf = os.path.join(r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\morphometrics_w_projs\classification\wrong_classifications", cell+'.png')
        fig.savefig(sf)
        plt.close()
    
#%% look at some trees idk
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree, feature_names = Ffeat.columns, filled=True, impurity=True, proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
    
#%% feature importances

importances = pd.Series(rf.feature_importances_, index=Ffeat.columns)
importances.sort_values(ascending=False, inplace=True)
#%%plot
plt.bar(importances.head(20).index, importances.head(20).values)

plt.tick_params(axis='x', rotation=90)
#%%
imp_ccf = importances.filter(regex=r' length| ends')

sums = defaultdict(int)
idx = []
for col, value in imp_ccf.items():
    reg = col.split(' ')[0]
    sums[reg] += value
    idx.append(reg + ' length+ends')

fig, ax = plt.subplots(figsize=(20,10))
ax.bar(sums.keys(), sums.values())
plt.tick_params(axis='x', rotation=90, labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.tight_layout()

#%% integrate summed and morphometric gini reduction

ccf_ser = pd.Series(sums)

imp_drop = importances.drop(imp_ccf.index)
imp_sum = pd.concat([imp_drop, ccf_ser])
imp_sum.sort_values(ascending=False, inplace=True)

fig2, ax2 = plt.subplots(figsize=(50,10))
ax2.bar(imp_sum.index, imp_sum.values)
plt.tick_params(axis='x', rotation=90, labelsize=18)
plt.tight_layout()

#%% find regions to drop

drop = imp_sum[imp_sum<0.0031].index

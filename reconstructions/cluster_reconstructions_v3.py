"""
alright heres v3 of my clustering attempts, this one I plan to ln scale my data first then cluster using hierarchical clustering with wards linkage, as 
I've seen in a bunch of papers
"""
import imagej
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

def drop_by_thresh(df, thresh):
    '''
    this is written to drop columns of a data frame that have a sum value of below the input threshold
    df: pandas.DataFrame
    thresh: float
    returns: pandas.DataFrame
    '''
    sum_df = df.apply(np.sum, axis=0)
    mask = sum_df < thresh
    low_endpoint_regions = sum_df[mask].index
    return df.drop(low_endpoint_regions, axis=1)

ij = imagej.init(ij_dir_or_version_or_endpoint=r'C:\Users\samkr\OneDrive\Desktop\fiji-win64 (2)\Fiji.app\fiji-windows-x64.exe', mode='Mode.GUI')
#ij.run_plugin

frequency_dir = r"C:\Code\reconstructions\SNT_histos\SNT_histos\frequency"
length_dir = r"C:\Code\reconstructions\SNT_histos\SNT_histos\length"

data = pd.DataFrame()

#format data from each neuron correctly
print('Loading files:')
for file in tqdm(os.listdir(frequency_dir)):
    filename = os.path.join(frequency_dir, file)
    df = pd.read_csv(filename, header=None)
    df[0] = df[0] + df[1]
    df[1] = df[2]
    del df[2]
    df = df.loc[(df!=0).any(axis=1)]
    df_t = df.transpose()
    file = file.replace('_freq.csv','')
    df_t = df_t.rename(index={1: file})
    df_t.columns = df_t.iloc[0]
    df_t = df_t.drop([0])
    data = pd.concat([data, df_t], join='outer')
data_nonan = data.replace(np.nan, 0)
#print("Column variances:", np.var(data, axis=0))
#remove zero variance columns
selector = VarianceThreshold(threshold=0.0).set_output(transform='pandas')
data_nonan = selector.fit_transform(data_nonan)

#filter out regions that have less than 3 endpoints labeled in that region
data_thresh = drop_by_thresh(data_nonan, 3)
data_toln = data_thresh.copy()

#natural log scale my data, formula of ln(number of terminals + 1) as per Ding et al. 2025
for i, row in data_toln.iterrows():
    data_toln.loc[i] = row.map(lambda x: np.log(x+1))

# scipy.cluster.hierarchy.linkage clustering
# data_linkage = linkage(data_toln, method='ward', metric='euclidean')
# dendrogram(data_linkage, labels=data_toln.index)
# plt.show()



# seaborn hierarchy clustering
data_tocluster = data_toln.T
cmap = sns.clustermap(data_tocluster, 
                      method='ward', 
                      metric='euclidean', 
                      row_cluster=False,
                      dendrogram_ratio=(.1, .15),
                      cbar_pos=(0, .15, .03, .7),
                      cbar_kws={'label': 'ln(terminals in region + 1)'},
                      yticklabels=True,
                      xticklabels=True)
# cmap.ax
# reordered_row_indices = cmap.dendrogram_row.reordered_ind
# print(reordered_row_indices)
plt.show()
'''
think im going to have to put a hold on this as running pca on the df of endpoints in each region is returning pcs that explain an annoyingly low amount of variance per pc,
ill have to write a script to interact with snt and get out the data i need i think
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import os
import traceback
from tqdm import tqdm

#first, writing all functions I need at the top
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

def scale_as_desired(df, use_scaler):
    '''
    this is written to scale a dataframe using the desired scaler, currently only functional for the MinMaxScaler and the StandardScaler in sklearn.preprocessing
    df: pandas.DataFrame
    use_scaler: string, accepted values are 'minmax' and 'standard'
    returns: pandas.DataFrame
    '''
    match use_scaler:
        case 'minmax':
            scaler_mm = MinMaxScaler()
            scaled_df_mm = scaler_mm.fit_transform(df)
            return scaled_df_mm
        case 'standard':
            scaler_ss = StandardScaler()
            scaled_df_ss = scaler_ss.fit_transform(df)
            return scaled_df_ss
        case _:
            raise ValueError("use_scaler can only be 'minmax' or 'standard'")
        
def run_pca(df):
    '''
    fit and run pca on a pandas dataframe
    returns the transformed dataframe along with the PCA object
    '''
    pca = PCA(n_components=0.9, svd_solver='full')
    pca.fit(df)
    transform_df = pca.transform(df)
    return transform_df, pca

# cmd = 'conda activate reconstructions'
# os.system(cmd)

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
#pd.set_option('future.no_silent_downcasting', True)
data_nonan = data.replace(np.nan, 0)
#print("Column variances:", np.var(data, axis=0))
#remove zero variance columns
selector = VarianceThreshold(threshold=0.0).set_output(transform='pandas')
data_reduced = selector.fit_transform(data_nonan)

# some visualization/quick analysis first to see how my data looks, help decide how i want to normalize (will be empty when done)
# sum_df = data_reduced.apply(np.sum, axis=0)
# print(f'Total endpoints of each cell: {sum_df}')
# print(type(sum_df))

#just arbitrarily picking some thresholds right now, not exactly sure what to use as i didn't find people who did this
#filtering out regions that have less than 5, 10, or 15 labeled endpoints
data_thresh5 = drop_by_thresh(data_reduced, 5)
data_thresh10 = drop_by_thresh(data_reduced, 10)
data_thresh15 = drop_by_thresh(data_reduced, 15)

#check for drop_by_thresh
# sum_5 = data_thresh5.apply(np.sum, axis=0)
# mask = sum_5 < 5
# empty = sum_5[mask].index

# to_drop = []
# for thresh in threshs:
    
#     print(f'regions with less than {thresh} endpoints: {low_endpoint_regions}')

#plt.bar(sum_df)


#normalize
data5_mm = scale_as_desired(data_thresh5, 'minmax')
data5_ss = scale_as_desired(data_thresh5,'standard')
data10_mm = scale_as_desired(data_thresh10, 'minmax')
data10_ss = scale_as_desired(data_thresh10,'standard')
data15_ss = scale_as_desired(data_thresh15,'standard')
data15_mm = scale_as_desired(data_thresh15, 'minmax')

full_data_list = [data5_mm, data5_ss, data10_mm, data10_ss, data15_mm, data15_ss]

#PCA
data5_mm_trans, data5_mm_PCA = run_pca(data5_mm)
print(data5_mm_PCA.explained_variance_ratio_, sum(data5_mm_PCA.explained_variance_ratio_))
#columns = data.keys()
# norm_data = data_reduced.copy()
# mm_scaler = MinMaxScaler()
# Sscaler = StandardScaler()
# norm_data_mm = mm_scaler.fit_transform(norm_data)
# norm_data_z = Sscaler.fit_transform(norm_data)
#print(f"MinMaxScaler scaled: {norm_data_mm}")
#print(f"StandardScaler scaled: {norm_data_z}")
#print(type(norm_data_z))
#for i, row in norm_data.iterrows():

    #norm_data[column] = scaler.fit_transform(np.array(norm_data[column]).reshape(-1,1))
    #print(row)

# checking for zero variance columns (StandardScaler doesn't scale columns with zero variance, so they would get a .scale_ value of 1)    
# zero_var_idx = [i for i, val in enumerate(Sscaler.scale_) if val == 1] 
# print(zero_var_idx)



#norm_data = norm_data.loc[(norm_data!=0).any(axis=1)]
#print('normalized data:', norm_data.head())
#norm_np = norm_data.to_numpy()
# norm_bool = norm_np.any(axis=1)
# for i, entry in enumerate(norm_bool):
#     if entry is False:
#         print(f"0 row found at index: {i}")
#     else:
#         print(f"row {i} has all non 0 values")

#PCA
#pca = PCA(n_components = .9, svd_solver = 'full') 
# try:
#     pca.fit(norm_np_reduced)
#     PCAdata = pca.transform(norm_np_reduced)
# except Exception as e:
#     print("PCA failed, exception:")
#     traceback.print_exc()
# print('PCA complete, data proj onto PCs:', norm_np_reduced)
# exp_var = pca.explained_variance_ratio_
# cols = ["PC"+str(i+1) for i in range(PCs.shape[1])]
# principalDF = pd.DataFrame(data=PCs, columns=cols)
# print("PCs: ", principalDF)
# print(exp_var, sum(exp_var))
# print(PCAdata, len(PCAdata))

#clustering

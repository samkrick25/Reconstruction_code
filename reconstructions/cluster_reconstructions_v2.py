"""
v2 but i think this is also going to fail, can come back and use these measurements if i want to later, writing a new file for v3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

def main():
    file = r"C:\Reconstructions\registered_MY_IRN_PARN_PGRN\measurements\reg_recon_measurements.csv"
    columns = ['Cable length', 'Depth', 'Height', 'No. of branch points', 'No. of primary branches', 'No. of tips', 'No. of total nodes', 'Volume', 'Width', 'No. of compartments']

    data = pd.read_csv(file, delimiter=',', names=columns)
    #print(data.head())
    pca = PCA()
    pca.fit(data)
    transform_data = pca.transform(data)
    print(pca.explained_variance_ratio_)

if __name__ == '__main__':
    main()

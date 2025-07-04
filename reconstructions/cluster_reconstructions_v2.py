import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

def main():
    file = r"C:\Data\reconstructions\medulla_IRN_PRN_PGRN\swc_measurements\reg_recon_measurements.csv"
    columns = ['Cable length', 'Depth', 'Height', 'No. of branch points', 'No. of primary branches', 'No. of tips', 'No. of total nodes', 'Volume', 'Width', 'No. of compartments']

    data = pd.read_csv(file, delimiter=',', names=columns)
    #print(data.head())
    pca = PCA()
    pca.fit(data)
    transform_data = pca.transform(data)
    print(pca.explained_variance_ratio_)

if __name__ == '__main__':
    main()

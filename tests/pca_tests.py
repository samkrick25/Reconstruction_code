import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
# cmd = 'conda activate reconstructions'
# os.system(cmd)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# print('pca object created')
# pca.fit(X)
# PCA(n_components=2)
# print('pca complete')
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
df = pd.DataFrame({
    'col1': [1, 3],
    'col2': [3,5]
})
for row in df[:]:
    print(df[row])
#print(df['col1'])
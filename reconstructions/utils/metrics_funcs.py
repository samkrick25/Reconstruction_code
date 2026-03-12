import numpy as np
import pandas as pd
from reconstructions.utils import preprocess_funcs as pp
import pickle
from reconstructions.utils.filedirs import freqspkl

def lat_index(df, region):
    '''
    find laterality index of cells in frequency dataset, lat index is (%proj ipsilateral - %proj contralateral)/100
    -1 is solely contralat projecting, 1 is solely ipsilateral projecting, 0 is perfectly bilaterally projecting

    :params df: DataFrame containing projection frequencies in shape (n cells, n regions)

    :params region: target region to calc lat index for as str, currently this is only accepting one region at a time
    '''
    data = pp.get_df_for_region(df, region=region)
    cols = list(data.columns)
    sums = data.sum(axis=1)
    pct = (data.div(sums, axis=0))*100
    pct['Laterality Index'] = (pct[cols[0]]-pct[cols[1]])/100

    return pct

def get_cell_lateralization(df):
    '''
    gets lists of cells that are lateralized for a certain region, requires output of lat_index

    :params df: DataFrame that is lateralization info, output of lat_index

    returns: 3 lists that are ordered ipsi, contra, bilat
    '''
    ipsi = []
    contra = []
    bilat = []

    for name, row in df.iterrows():
        lati = row["Laterality Index"]
        if lati > 0.75:
            ipsi.append(name)
        if lati < -0.75:
            contra.append(name)
        if -0.75 <= lati <= 0.75:
            bilat.append(name)

    return ipsi, contra, bilat

def get_cells_to_region(df, region):
    '''
    get cells that project to a target region
    
    :param df: DataFrame containing laterality merged frequency data

    :param region: str of target region

    returns: list containing names of all cells that have nonzero endpoint frequencies in a target region
    '''
    cells = []
    
    values = df[region]
    nozero = values[values!=0].index.to_list()

    return nozero

def get_targets(df, cell):
    targetrow = df.loc[cell]
    targets = targetrow[targetrow != 0].index.tolist()
    return targets

if __name__=='__main__':
    freqs = pickle.load(open(freqspkl, 'rb'))
    targetcell = 'AA0503'
    targets = get_targets(freqs, targetcell)
    print(targets)
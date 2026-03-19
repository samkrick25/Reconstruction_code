import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_selection import VarianceThreshold
from warnings import simplefilter

MIDLINEZ = 5750
MIDLINEZ_10UM = 570

def rem_zero_var(df):
    '''
    removes columns with 0 variance from a df, might not be necessary if my json reading is working correctly
    
    :param df: DataFrame to have colums dropped from
    '''
    selector = VarianceThreshold(threshold=0.0).set_output(transform='pandas')
    df = selector.fit_transform(df)
    return df


def preprocess(df, log1p=True, pct=True):
    '''
    preprocess data however I want to, accepts a pandas.DataFrame with no NaNs

    :param df: pandas.DataFrame

    :param log1p: preprocessing option, default True, if False skips log1p scaling

    :param pct: preprocessing option, default True, if False skips cell size scaling

    returns: pandas.DataFrame
    '''
    data_tonorm = df.copy()

    #natural log scale my data, formula of ln(number of terminals + 1) as per Ding et al. 2025
    #and control for cell size
    #default behavior
    if log1p and pct:
        df_log = np.log(data_tonorm+1)
        row_sums = df_log.sum(axis=1)
        df_pct = (df_log.div(row_sums, axis=0))*100
        return df_pct
    
    if log1p and not pct:
        df_log = np.log(data_tonorm+1)
        return df_log
    
    if not log1p  and pct:
        row_sums = df.sum(axis=1)
        df_pct = (df.div(row_sums, axis=0))*100
        return df_pct
    
    if not log1p and not pct:
        print('what are you using this for')
        return
    
def merge_regions(df):
    '''
    merge ipsilateral/contralateral regions into one column
    (GPT assist)

    :param df: DataFrame with frequency information for cells, columns must be 'Ipsilateral [region]' or 'Contralateral [region]'

    returns: DataFrame with shape (n cells, n regions)
    '''
    #this suppresses the PerformanceWarning that pd throws, this function is still running fast
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    ipsi_cols = pd.Series(col for col in df.columns if col.startswith('Ipsilateral'))
    contra_cols = pd.Series(col for col in df.columns if col.startswith('Contralateral'))
    ipsi_regions = ipsi_cols.str.replace('Ipsilateral ', '')
    contra_regions = contra_cols.str.replace('Contrlateral ', '')
    all_regions = set(ipsi_regions) | set(contra_regions)
    
    merged_frequency = pd.DataFrame(index=df.index)
    for region in all_regions:
        ipsi_col = f'Ipsilateral {region}'
        contra_col = f'Contralateral {region}'
        ipsi_series = df[ipsi_col] if ipsi_col in df.columns else None
        contra_series = df[contra_col] if contra_col in df.columns else None
        
        #regions that recieve both ipsi and contra projections
        if ipsi_series is not None and contra_series is not None:
            merged_frequency[region] = df[ipsi_col] + df[contra_col]
            
        #regions that recieve only ipsilateral
        if ipsi_series is not None and contra_series is None:
            merged_frequency[region] = df[ipsi_col]
            
        #regions that only recieve contra
        if ipsi_series is None and contra_series is not None:
            merged_frequency[region] = df[contra_col]
    
    return merged_frequency

def get_df_for_region(df, region):
    '''
    get a df containing lateralized frequency data for a given target region

    :params df: full dataframe with columns as regions, lateralized, and rows as neurons

    :params region: target region as str
    '''
    ipsireg = 'Ipsilateral '+region
    contrareg = 'Contralateral '+region
    fullreg = [ipsireg, contrareg]

    regdf = df[fullreg]
    for cell, val in regdf.iterrows():
        iv, cv = val
        if iv == 0 and cv == 0:
            regdf = regdf.drop(cell)

    return regdf

def get_nodes_in_region(cells, *regions, kind=None):
    '''
    Docstring for get_nodes_in_region
    
    :param cells: Description
    :param regions: Description
    '''
    match kind:
        case 'bulk':
            nodes = []
            for _, axon in cells.items():
                for node in axon:
                    if node['allenId'] in regions:
                        nodes.append(node)
            return nodes
        case 'by_cell':
            cellstonodes = {}
            for cell, axon in cells.items():
                nodes = []
                for node in axon:
                    if node['allenId'] in regions:
                        nodes.append(node)
                cellstonodes[cell] = nodes
            return cellstonodes
        
def get_target_nodes_list(nodes, *regions):
    '''
    Docstring for get_target_nodes_list
    
    :param nodes: Description
    :param regions: Description
    '''
    targets = []
    for node in nodes:
        if node['allenId'] in regions:
            targets.append(node)
    return targets

def get_coords(nodes, dim='all', mirror=False):
    """
    Coordinate getter for a list of nodes contianing x, y, z coordinates
    
    :param nodes: list of dictionaries, each entry should be a dictionary containing 'x', 'y', 'z' coordinates for said node
    :param dim: str, selecting which coordinates to get, default behavior is to return all coords
    :param mirror: bool, default False to not mirror nodes, but will mirror nodes over saggital axis (z in ccfv3)

    Returns: list of coordinates if only one dimension is selected, np.array of lists where each list contains x, y, z coords if all dims are selected
    """
    match dim:
        case 'x':
            x = [node['x'] for node in nodes]
            return x
        case 'y':
            y = [node['y'] for node in nodes]
            return y
        case 'z':
            if mirror:
                for node in nodes:
                    if node['z'] < 5700:
                        diff = 5700-node['z']
                        node['z'] = diff
                z = [node['z'] for node in nodes]
            else:
                z = [node['z'] for node in nodes]            
            return z
        case 'all':
            for node in nodes:
                if mirror:
                    if node['z'] < 5700:
                        diff = 5700 - node['z']
                        node['z'] = 5700+diff
            coords = np.array([[node['x'], node['y'], node['z']] for node in nodes]) 
            return coords
        
def node_coords_getter(cell, dim, *regions):
    targets = get_target_nodes_list(cell, regions)
    coords = np.array(get_coords(cell, dim) for cell in targets if cell)
    return coords

def get_cells_to_region(freqspkl, regionabv, thresh=3):
    freqs = pickle.load(open(freqspkl, 'rb'))

    latmerged = merge_regions(freqs)
    poscells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell][regionabv] > thresh]
    negcells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell][regionabv] <= thresh]

    return poscells, negcells

def get_freqs(neuronsdict, aidtoreg):
    columns = []
    for _, (_, abv) in aidtoreg.items():
        ipsi = 'Ipsilateral ' + abv
        contra = 'Contralateral ' + abv
        columns.append(ipsi)
        columns.append(contra)
    freqdf = pd.DataFrame(columns=columns)
    for neuron, axon in neuronsdict.items():
        ...
        

    
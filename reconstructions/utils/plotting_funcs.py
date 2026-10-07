import matplotlib.pyplot as plt
import seaborn as sns
from reconstructions.utils import preprocess_funcs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.decomposition import SparsePCA
import pandas as pd
import os
import umap


def comp_node_dist(points1, points2, suptitle=None, labels=None, ver=3.0, colors=['red', 'blue']):
    '''
    Compare node distribution in x, y, z coordinates of two populations of points
    
    :param points1: list of dictionaries, nodes from a reconstruction's json, to be input into preprocess_funcs.get_coords
    :param points2: list of dictionaries, nodes from a reconstruciton's json, to be input into preprocess_funcs.get_coords
    :param suptitle: str, input to fig.suptitle
    :param labels: list of str, labels for the two populations to compare
    :param ver: float, allen ccf version, to be used for x labels, either 3.0 or 2.5, written as x and z were switched between two versions
    :param colors: array of str, the colors you want the two populations to be labeled as, default blue and red
    '''
    points1x = preprocess_funcs.get_coords(points1, 'x')
    points1y = preprocess_funcs.get_coords(points1, 'y')
    points1z = preprocess_funcs.get_coords(points1, 'z')

    points2x = preprocess_funcs.get_coords(points2, 'x')
    points2y = preprocess_funcs.get_coords(points2, 'y')
    points2z = preprocess_funcs.get_coords(points2, 'z')

    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    xax, yax, zax = axes
    if suptitle:
        fig.suptitle(suptitle)
    fig.supylabel('kernel density')
    fig.supxlabel('allen CCF coordinates')
    
    match ver:
        case 3.0:
            xax.set_xlabel('AP')
            yax.set_xlabel('DV')
            zax.set_xlabel('ML')
        case 2.5:
            xax.set_xlabel('ML')
            yax.set_xlabel('DV')
            zax.set_xlabel('AP')

    sns.kdeplot(data=points1x, color=colors[0], alpha=0.5, ax=xax)
    sns.kdeplot(data=points2x, color=colors[1], alpha=0.5, ax=xax)

    sns.kdeplot(data=points1y, color=colors[0], alpha=0.5, ax=yax)
    sns.kdeplot(data=points2y, color=colors[1], alpha=0.5, ax=yax)

    sns.kdeplot(data=points1z, color=colors[0], alpha=0.5, ax=zax)
    sns.kdeplot(data=points2z, color=colors[1], alpha=0.5, ax=zax)

    if labels:
        lines = xax.get_lines()
        lines[0].set_label(labels[0])
        lines[1].set_label(labels[1])
        fig.legend()

    return fig, axes
# =============================================================================
# 
# def comp_node_dist(*dataargs, suptitle=None, labels=None, ver=3.0, colors=['blue', 'red']):
#     '''
#     Compare node distribution in x, y, z coordinates of two populations of points
#     
#     :param points1: list of dictionaries, nodes from a reconstruction's json, to be input into preprocess_funcs.get_coords
#     :param points2: list of dictionaries, nodes from a reconstruciton's json, to be input into preprocess_funcs.get_coords
#     :param suptitle: str, input to fig.suptitle
#     :param labels: list of str, labels for the two populations to compare
#     :param ver: float, allen ccf version, to be used for x labels, either 3.0 or 2.5, written as x and z were switched between two versions
#     :param colors: array of str, the colors you want the two populations to be labeled as, default blue and red
#     '''
#     datax = [preprocess_funcs.get_coords(data, 'x') for data in dataargs]
#     datay = [preprocess_funcs.get_coords(data, 'y') for data in dataargs]
#     dataz = [preprocess_funcs.get_coords(data, 'z') for data in dataargs]
# 
#     fig, (xax, yax, zax) = plt.subplots(1, 3, figsize=(20,6))
#     if suptitle:
#         fig.suptitle(suptitle)
#     fig.supylabel('kernel density')
#     fig.supxlabel('allen CCF coordinates')
#     
#     match ver:
#         case 3.0:
#             xax.set_xlabel('AP')
#             yax.set_xlabel('DV')
#             zax.set_xlabel('ML')
#         case 2.5:
#             xax.set_xlabel('ML')
#             yax.set_xlabel('DV')
#             zax.set_xlabel('AP')
#     
#     for i, data in enumerate(datax):
#         sns.kdeplot(data=data, color=colors[i-1], alpha=0.5, ax=xax)
#     
#     for i, data in enumerate(datay):
#         sns.kdeplot(data=data, color=colors[i-1], alpha=0.5, ax=yax)
#         
#     for i, data in enumerate(dataz):
#         sns.kdeplot(data=data, color=colors[i-1], alpha=0.5, ax=zax)
# 
#     if labels:
#         lines = xax.get_lines()
#         for i in len(lines):
#             lines[i-1].set_label(labels[i-1])
#         
#         fig.legend()
# 
#     return fig
# =============================================================================

def plot_node_dist(nodes, colors=['blue']):
    '''
    plot distribution of nodes in AP, ML, DV axes

    Parameters
    ----------
    nodes : list
        list of lists where inner lists are [x, y, z] coordinates of each node.
    colors : list, optional
        list of colors for lines to have on plots. The default is ['blue'].
        right now this only plots lines as one color

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and Axes objects containing plot info.

    '''
    pointsx = preprocess_funcs.get_coords(nodes, 'x')
    pointsy = preprocess_funcs.get_coords(nodes, 'y')
    pointsz = preprocess_funcs.get_coords(nodes, 'z', mirror=True)
    fig, (xax, yax, zax) = plt.subplots(1, 3, figsize=(20,6))
    xax.set_xlabel('AP')
    yax.set_xlabel('DV')
    zax.set_xlabel('ML')
    sns.kdeplot(pointsx, color=colors[0], alpha=0.5, ax=xax)
    sns.kdeplot(pointsy, color=colors[0], alpha=0.5, ax=yax)
    sns.kdeplot(pointsz, color=colors[0], alpha=0.5, ax=zax)
    return fig, (xax, yax, zax)

def do_pca_and_plot(data, sd, n_components=None, plot=False):
    '''
    do pca on your data, returns pca object as well as transformed data, also plots loadings onto the top 10 pcs

    Parameters
    ----------
    data : array-like
        data to do PCA on
    sf : str
        desired save location, directory
    n_components : int
        desired number of components for PCA to find
    Returns
    -------
    PCA object and transformed data.

    '''
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    if n_components:
        pca=PCA(n_components=n_components)
    else:
        pca=PCA()
    
    pca.fit(scaled)
    scaled = pca.fit_transform(scaled)
    

    loadings = pd.DataFrame(pca.components_, columns = data.columns, index = [f'PC{i+1}' for i in range(pca.n_components_)])

    def get_top_loading(df, component, n=15):
        pc = df.loc[component]
        
        return pc.reindex(pc.abs().sort_values(ascending=False).index).head(n)
    
    if plot:
        for i in range(1,n_components+1 if n_components else 11):
            string = 'PC'+str(i)
            l = get_top_loading(loadings, string)
            plt.bar(l.index, l.values)
            plt.title('Loadings onto '+string)
            plt.tick_params(axis='x', labelrotation=90)
            plt.tight_layout()
            sf = os.path.join(sd,string+'.png')
            plt.savefig(sf)
            plt.close()
            
    return pca, scaled, loadings

def do_spca_and_plot(data, alpha, n_components=None, sd=None, plot=False):
    '''
    do pca on your data, returns pca object as well as transformed data, also plots loadings onto the top 10 pcs

    Parameters
    ----------
    data : pd.DataFrame
        data to do PCA on
    sf : str
        desired save location, directory
    n_components : int
        desired number of components for PCA to find
    Returns
    -------
    PCA object and transformed data stored as DataFrame.

    '''
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    if n_components:
        pca=SparsePCA(alpha = alpha, n_components=n_components)
    else:
        pca=SparsePCA(alpha=alpha)
    
    pca.fit(scaled)
    scaled = pca.fit_transform(scaled)
    
    scaled = pd.DataFrame(data=scaled, index=data.index)
    
    loadings = pd.DataFrame(pca.components_, columns = data.columns, index = [f'PC{i+1}' for i in range(pca.n_components_)])

    def get_top_loading(df, component, n=15):
        pc = df.loc[component]
        
        return pc.reindex(pc.abs().sort_values(ascending=False).index).head(n)

    if plot:
        for i in range(1,n_components+1):
            string = 'PC'+str(i)
            l = get_top_loading(loadings, string)
            plt.bar(l.index, l.values)
            plt.title('Loadings onto '+string)
            plt.tick_params(axis='x', labelrotation=90)
            plt.tight_layout()
            sf = os.path.join(sd,string+'.png')
            plt.savefig(sf)
            plt.close()
        
    return pca, scaled

def umap_and_plot(pcadata, index, sd=None, colors=None, **kwargs):
    '''
    UMAP data and scatter plot

    Parameters
    ----------
    pcadata : ndarray
        data already dimensionality reduced by PCA.
    index : pd.Index
        index of the data you are plotting
    sd : str
        path that you want to save UMAP plot to, default None/not saved
    colors : list
        list of colors to be passed to matplotlib, default None/use matplotlib default coloring
    **kwargs : kwargs
        keyword arguments to pass to UMAP fit_transform().

    Returns
    -------
    UMAP embedding DataFrame.

    '''
    
    reducer = umap.UMAP(**kwargs)

    scaled_sPCA_forUMAP = StandardScaler().fit_transform(pcadata)

    embedding = reducer.fit_transform(scaled_sPCA_forUMAP)
    
    embeddingdf = pd.DataFrame(embedding, index=index, columns = ['UMAP1', 'UMAP2'])

    plt.scatter(embeddingdf['UMAP1'], embeddingdf['UMAP2'], c=colors, alpha=0.7)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.xticks([])
    plt.yticks([])
    if sd:
        plt.savefig(sd)
    
    return embeddingdf
import matplotlib.pyplot as plt
import seaborn as sns
from reconstructions.utils import preprocess_funcs

def comp_node_dist(points1, points2, suptitle=None, labels=None, ver=3.0, colors=['blue', 'red']):
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

    fig, (xax, yax, zax) = plt.subplots(1, 3, figsize=(20,6))
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

    return fig

def plot_node_dist(nodes, colors=['blue']):
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
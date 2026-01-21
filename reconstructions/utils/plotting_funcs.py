import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_funcs

def plot_axon_dist_in_region(upper1nodes, upper2nodes, region):
    upper1x = preprocess_funcs.get_coords(upper1nodes, 'x')
    upper1y = preprocess_funcs.get_coords(upper1nodes, 'y')
    upper1z = preprocess_funcs.get_coords(upper1nodes, 'z')

    upper2x = preprocess_funcs.get_coords(upper2nodes, 'x')
    upper2y = preprocess_funcs.get_coords(upper2nodes, 'y')
    upper2z = preprocess_funcs.get_coords(upper2nodes, 'z')

    fig, (xax, yax, zax) = plt.subplots(1, 3, figsize=(20,6))
    fig.suptitle(f'PT distribution in {region}')
    fig.supylabel('kernel density')
    fig.supxlabel('allen CCF coordinates')

    xax.set_xlabel('ML')
    yax.set_xlabel('DV')
    zax.set_xlabel('AP')

    sns.kdeplot(data=upper1x, color='blue', alpha=0.5, ax=xax)
    sns.kdeplot(data=upper2x, color='red', alpha=0.5, ax=xax)

    sns.kdeplot(data=upper1y, color='blue', alpha=0.5, ax=yax)
    sns.kdeplot(data=upper2y, color='red', alpha=0.5, ax=yax)

    sns.kdeplot(data=upper1z, color='blue', alpha=0.5, ax=zax)
    sns.kdeplot(data=upper2z, color='red', alpha=0.5, ax=zax)

    lines = xax.get_lines()
    lines[0].set_label('PT_upper 1')
    lines[1].set_label('PT_upper 2')
    fig.legend()

    return


from reconstructions.utils import preprocess_funcs, plotting_funcs
from reconstructions.utils.cameras import topcam, sagcam, corcam
from reconstructions.utils.filedirs import freqspkl, somaspkl
from reconstructions.utils.planes import medplane
import numpy as np
import pickle
from brainrender.actors import Points
from brainrender import Scene
import vedo

freqs = pickle.load(open(freqspkl, 'rb'))
somas = pickle.load(open(somaspkl, 'rb'))

latmerged = preprocess_funcs.merge_regions(freqs)
GRNposcells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell]['GRN'] > 3]
GRNnegcells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell]['GRN'] <= 3]

#get mirrored coordinates of GRN positive and GRN negative somas
GRNpossomas = [soma for cell, soma in somas.items() if cell in GRNposcells]
GRNposnames = [cell for cell, _ in somas.items() if cell in GRNposcells]
GRNpospoints = preprocess_funcs.get_coords(GRNpossomas, mirror=True)
GRNnegsomas = [soma for cell, soma in somas.items() if cell in GRNnegcells]
GRNnegnames = [cell for cell, _ in somas.items() if cell in GRNnegcells]
GRNnegpoints = preprocess_funcs.get_coords(GRNnegsomas, mirror=True)
print(f'GRN positive cells (terminals > 3): {GRNposnames}, ncells:{len(GRNposnames)}')
print(f'GRN negative cells (terminals <= 3): {GRNnegnames}, ncells:{len(GRNnegnames)}')
print(f'total cells: {len(GRNnegnames) + len(GRNposnames)}')


ccf_scene = Scene(root=False, atlas_name='allen_mouse_10um')
ccf_scene.add_brain_region('IRN', color='red', alpha=0.05, silhouette=False)
ccf_scene.add_brain_region('PARN', color='blue', alpha=0.05, silhouette=False)

GRNpospoints = Points(GRNpospoints, colors='green', name='GRN positive somas')
GRNnegpoints = Points(GRNnegpoints, colors='purple', name='GRN negative somas')
ccf_scene.add(GRNpospoints)
ccf_scene.add(GRNnegpoints)

ccf_scene.slice(plane=medplane)
ccf_scene.render(camera=corcam)

axondistfig = plotting_funcs.comp_node_dist(GRNpossomas, GRNnegsomas, suptitle='Soma Distribution of GRN+ and GRN- cells',
                                            labels=['GRN positive somas', 'GRN negative somas'], colors=['green', 'purple'])
savepath = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\plots\GRNvnonGRNsomadist.png'
axondistfig.savefig(savepath, dpi=300, bbox_inches='tight')
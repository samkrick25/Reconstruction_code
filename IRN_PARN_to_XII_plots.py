from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import plotting_funcs as pl
from reconstructions.utils.filedirs import freqspkl, somaspkl, allcoordswapped, parcellated_neurons
import pickle
from tqdm import tqdm
import os
import numpy as np
from reconstructions.utils.meshAnalyzer import MeshCoordinateSystem
import seaborn as sns
import matplotlib.pyplot as plt

#os.system('conda activate reconstructions')

freqs = pickle.load(open(freqspkl, 'rb'))
somas = pickle.load(open(somaspkl, 'rb'))

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=False)
ccf_scene.add_brain_region('XII', color='orange', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('IRN', color='red', alpha=0.0, silhouette=False)
ccf_scene.add_brain_region('PARN', color='pink', alpha=0.0, silhouette=False)

actors = ccf_scene.get_actors()
XIImesh = actors[0]
IRNmesh = actors[1]
PARNmesh = actors[2]
XIIvertices = XIImesh.mesh.vertices
# =============================================================================
# mirroredMeshverts = []
# for vertex in XIIvertices:
#     if vertex[2] > 5682.260401:
#         mirroredMeshverts.append(vertex)
# XIImesh.mesh.vertices = mirroredMeshverts
# =============================================================================
XIIverticesmirrored = XIImesh.mesh.vertices
IRNvertices = IRNmesh.mesh.vertices
PARNvertices = PARNmesh.mesh.vertices

IRNap = [vertex[0] for vertex in IRNvertices]
PARNap = [vertex[0] for vertex in IRNvertices]
IRNPARNap = IRNap+PARNap
MedRNa_bound = np.min(IRNPARNap)
MedRNp_bound = np.max(IRNPARNap)
MedRNrange = MedRNp_bound-MedRNa_bound
MedRNmid = MedRNa_bound + 800
#MedRNmid2 = MedRNa_bound + 2*(MedRNmid1)

antIPARNcells = []
postIPARNcells = []
for cell, soma in somas.items():
    if MedRNa_bound <= soma['x'] < MedRNmid:
        antIPARNcells.append(cell)
    if MedRNmid <= soma['x'] <= MedRNp_bound:
        postIPARNcells.append(cell)
        
antIPARNends = {}
postIPARNends = {}
for file in tqdm(os.listdir(parcellated_neurons), desc='Loading endpoints'):
    cellname = file.split('.')[0]
    endpoints = ld.get_endpoints_from_file_parcellated(os.path.join(parcellated_neurons, file))
    if cellname in antIPARNcells:
        antIPARNends[cellname] = endpoints
    if cellname in postIPARNcells:
        postIPARNends[cellname] = endpoints
        
antIPARNXII = pp.get_nodes_in_region(antIPARNends, 'XII', parcellated=True, kind='by_cell')
postIPARNXII = pp.get_nodes_in_region(postIPARNends, 'XII', parcellated=True, kind='by_cell')

antIPARNXIIswapped = pp.tenmicron_to_one(antIPARNXII, allcoordswapped)
postIPARNXIIswapped = pp.tenmicron_to_one(postIPARNXII, allcoordswapped)
#finish cleaning up from here later, need to rewrite this a little since above is now by cell, first need to look at coordswapped for actual coords
antIPARNXIIcoords = []
for cell, nodes in antIPARNXIIswapped.items():
    coords = pp.get_coords(nodes, mirror=True)
    if coords.any():
        antIPARNXIIcoords.append(coords)
antIPARNXIIflat = np.array([node for cell in antIPARNXIIcoords for node in cell])

postIPARNXIIcoords = []
for cell, nodes in postIPARNXIIswapped.items():
    coords = pp.get_coords(nodes, mirror=True)
    if coords.any():
        postIPARNXIIcoords.append(coords)
postIPARNXIIflat = np.array([node for cell in postIPARNXIIcoords for node in cell])

XIIsystem = MeshCoordinateSystem(XIIvertices)
antIPARNXII_transformed = XIIsystem.transform_to_local(antIPARNXIIflat)
postIPARNXII_transformed = XIIsystem.transform_to_local(postIPARNXIIflat) 

XIIap = [point[0] for point in XIIsystem.local_vertices]
XIIml = [point[1] for point in XIIsystem.local_vertices]
XIIdv = [point[2] for point in XIIsystem.local_vertices]

ant_x_transformed = [point[0] for point in antIPARNXII_transformed]
ant_y_transformed = [point[2] for point in antIPARNXII_transformed]
ant_z_transformed = [point[1] for point in antIPARNXII_transformed]

post_x_transformed = [point[0] for point in postIPARNXII_transformed]
post_y_transformed = [point[2] for point in postIPARNXII_transformed]
post_z_transformed = [point[1] for point in postIPARNXII_transformed]

colors=['red', 'blue']
fig, axes = plt.subplots(1, 3, figsize=(20,6))
xax, yax, zax = axes

sns.kdeplot(data=ant_x_transformed, color=colors[0], alpha=0.5, ax=xax)
sns.kdeplot(data=post_x_transformed, color=colors[1], alpha=0.5, ax=xax)

sns.kdeplot(data=ant_y_transformed, color=colors[0], alpha=0.5, ax=yax)
sns.kdeplot(data=post_y_transformed, color=colors[1], alpha=0.5, ax=yax)

sns.kdeplot(data=ant_z_transformed, color=colors[0], alpha=0.5, ax=zax)
sns.kdeplot(data=post_z_transformed, color=colors[1], alpha=0.5, ax=zax)

axes[0].axvline(x=np.max(XIIap))
axes[0].axvline(x=np.min(XIIap))
axes[1].axvline(x=np.max(XIIdv))
axes[1].axvline(x=np.min(XIIdv))
axes[2].axvline(x=np.max(XIIml))
#everything is mirrored into one hemisphere, so 
axes[2].axvline(x=0)

xax.set_xlabel('XII PC1')
yax.set_xlabel('XII PC2')
zax.set_xlabel('XII PC3')
fig.suptitle('Compartmental analysis of IRN/PARN projections to XII')
lines = xax.get_lines()
lines[0].set_label('Anterior IRN/PARN')
lines[1].set_label('Posterior IRN/PARN')
fig.legend()

fig.show()


from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
from vedo import Mesh

XIIfile = r'C:\Users\economolab\.brainglobe\allen_mouse_10um_v1.2\meshes\773.obj'

XIImesh = Mesh(XIIfile)
XIIvertices = XIImesh.vertices
# =============================================================================
# mirroredMeshverts = []
# for vertex in XIIvertices:
#     if vertex[2] > 5682.260401:
#         mirroredMeshverts.append(vertex)
# XIImesh.vertices = mirroredMeshverts
# XIIverticesmirrored = XIImesh.vertices
# =============================================================================

XIIorigin = np.mean(XIImesh.vertices, axis=0)
XIIcentered = XIIvertices-XIIorigin
pca = PCA(n_components=3)
pca.fit(XIIcentered)
XIItransformed = pca.transform(XIIcentered)
axes = pca.components_
x, y, z = np.array(XIItransformed).T
xc, yc, zc = np.array(XIIcentered).T

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, alphahull=0, color='lightpink', opacity=0.5)])
fig.add_trace(go.Mesh3d(x=xc, y=yc, z=zc, alphahull=0, color='lightblue', opacity=0.5))

pc1x = [0, axes[0][0]*100]
pc1y = [0, axes[0][1]*100]
pc1z = [0, axes[0][2]*100]
pc2x = [0, axes[1][0]*100]
pc2y = [0, axes[1][1]*100]
pc2z = [0, axes[1][2]*100]
pc3x = [0, axes[2][0]*100]
pc3y = [0, axes[2][1]*100]
pc3z = [0, axes[2][2]*100]

fig.add_trace(go.Scatter3d(x=pc1x, y=pc1y, z=pc1z, mode='lines', line=dict(color='blue', width=4), name="PC1"))
fig.add_trace(go.Scatter3d(x=pc2x, y=pc2y, z=pc2z, mode='lines', line=dict(color='green', width=4), name='PC2'))
fig.add_trace(go.Scatter3d(x=pc3x, y=pc3y, z=pc3z, mode='lines', line=dict(color='yellow', width=4), name='PC3'))

fig.show()
#terriergpt code below, going to read thru and make sense of everything before i continue
# =============================================================================
# # Load your data
# vertices = your_mesh_vertices  # Shape (n, 3)
# faces = your_mesh_faces  # Shape (m, 3), optional
# endpoints = your_endpoint_coordinates  # Shape (k, 3)
# 
# # Create analyzer
# analyzer = MeshDensityAnalyzer(vertices, faces)
# 
# # Analyze density along anterior-posterior axis (axis=0)
# results = analyzer.analyze_point_density(endpoints, axis=0, bin_size=10)
# 
# # Plot results
# fig = analyzer.plot_density_profile(results, axis=0, normalize=True)
# plt.show()
# 
# # Access transformed coordinates
# local_endpoints = results['local_points']
# 
# # You can also look at 2D distributions in local coordinates
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 
# # AP vs ML
# axes[0].scatter(local_endpoints[:, 0], local_endpoints[:, 1], 
#                alpha=0.5, s=1)
# axes[0].set_xlabel('Anterior-Posterior (μm)')
# axes[0].set_ylabel('Medial-Lateral (μm)')
# 
# # AP vs DV
# axes[1].scatter(local_endpoints[:, 0], local_endpoints[:, 2], 
#                alpha=0.5, s=1)
# axes[1].set_xlabel('Anterior-Posterior (μm)')
# axes[1].set_ylabel('Dorsal-Ventral (μm)')
# 
# # ML vs DV
# axes[2].scatter(local_endpoints[:, 1], local_endpoints[:, 2], 
#                alpha=0.5, s=1)
# axes[2].set_xlabel('Medial-Lateral (μm)')
# axes[2].set_ylabel('Dorsal-Ventral (μm)')
# 
# plt.tight_layout()
# plt.show()
# =============================================================================

XIIap = [vertex[0] for vertex in XIIsystem.local_vertices]
XIIdv = [vertex[1] for vertex in XIIsystem.local_vertices]
XIIlr = [vertex[2] for vertex in XIIsystem.local_vertices]
#input('Press Enter to continue')
# =============================================================================
# XIIprojcells = [cell for cell, row in freqs.iterrows() if row['Ipsilateral XII'] + row['Contralateral XII'] > 3]
# print(XIIprojcells)
# XIIends={}
# for file in tqdm(os.listdir(allcoordswapped), desc='Loading endpoints of XII+ cells'):
#     cellname = file.split('.')[0]
#     if cellname in XIIprojcells:
#         endpoints = ld.get_endpoints_from_file(os.path.join(allcoordswapped, file))
#         XIIends[cellname] = endpoints
# =============================================================================

#XIInodesbycell = pp.get_nodes_in_region(XIIends, 773, kind='by_cell')
#XIIcoords = ... #finish later, i can investigate this quesiton (topology of IRN/PARN->XII) by eye
#XIInodes = pp.get_nodes_in_region(XIIends, 773, kind='bulk')

#fig, axes = pl.plot_node_dist(XIInodes)
labels = ['anterior IRN/PARN','posterior IRN/PARN']
#this is expecting a list of nodes, not coordinates like I have right now, just rewrite later tbh
# =============================================================================
# fig, axes = pl.comp_node_dist(antIPARNXII_transformed, postIPARNXII_transformed, labels=labels, suptitle='Compartmental analysis of IRN/PARN projections to XII', colors=['red', 'blue'])
# axes[0].axvline(x=np.max(XIIap), label='XII posterior boundary')
# axes[0].axvline(x=np.min(XIIap), label='XII anterior boundary')
# axes[1].axvline(x=np.max(XIIdv), label="XII ventral boundary")
# axes[1].axvline(x=np.min(XIIdv), label="XII dorsal boundary")
# axes[2].axvline(x=np.max(XIIlr), label="XII lateral boundary")
# axes[2].axvline(x=np.min(XIIlr), label="XII medial boundary (midline)")
# fig.show()
# savepath = r'reconstructions\plots\antIRNPARNvpostIRNPARNXII.png'
# =============================================================================
#fig.savefig(savepath, dpi=300, bbox_inches='tight')
#input('Press Enter to continue')

#XIIcoords = pp.get_coords(XIInodes, dim='all')


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 02:27:43 2026

@author: economolab
"""

# =============================================================================
# from brainrender import Scene
# from brainrender.actors import Line
# from reconstructions.utils.meshAnalyzer import MeshCoordinateSystem
# from reconstructions.utils import cameras
# =============================================================================

from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
from vedo import Mesh

XIIfile = r'C:\Users\economolab\.brainglobe\allen_mouse_10um_v1.2\meshes\773.obj'
IRNfile = r'C:\Users\economolab\.brainglobe\allen_mouse_10um_v1.2\meshes\852.obj'
PARNfile = r'C:\Users\economolab\.brainglobe\allen_mouse_10um_v1.2\meshes\136.obj'

XIImesh = Mesh(XIIfile)
XIIvertices = XIImesh.vertices
mirroredMeshverts = []
for vertex in XIIvertices:
    if vertex[2] > 5682.260401:
        mirroredMeshverts.append(vertex)
XIImesh.vertices = mirroredMeshverts
XIIverticesmirrored = XIImesh.vertices

XIIorigin = np.mean(XIImesh.vertices, axis=0)
XIIcentered = XIIverticesmirrored-XIIorigin
pca = PCA(n_components=3)
pca.fit(XIIcentered)
XIItransformed = pca.transform(XIIcentered)
axes = pca.components_
xc, yc, zc = np.array(XIIcentered).T
x, y, z = np.array(XIItransformed).T

#fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, alphahull=0, color='lightpink', opacity=0.5)])
fig = go.Figure(data=[go.Mesh3d(x=xc, y=yc, z=zc, alphahull=0, color='lightblue', opacity=0.25)])

pc1x = [0, axes[0][0]*-500]
pc1y = [0, axes[0][1]*-500]
pc1z = [0, axes[0][2]*-500]
pc2x = [0, axes[2][0]*150]
pc2y = [0, axes[2][1]*150]
pc2z = [0, axes[2][2]*150]
pc3x = [0, axes[1][0]*200]
pc3y = [0, axes[1][1]*200]
pc3z = [0, axes[1][2]*200]

fig.add_trace(go.Scatter3d(x=pc1x, y=pc1y, z=pc1z, mode='lines', line=dict(color='blue', width=4), name="PC1"))
fig.add_trace(go.Scatter3d(x=pc2x, y=pc2y, z=pc2z, mode='lines', line=dict(color='green', width=4), name='PC2'))
fig.add_trace(go.Scatter3d(x=pc3x, y=pc3y, z=pc3z, mode='lines', line=dict(color='red', width=4), name='PC3'))

# =============================================================================
# XIIorigin = np.mean(XIIvertices, axis=0)
# XIIcentered = XIIvertices - XIIorigin
# pca = PCA(n_components=3)
# pca.fit(XIIcentered)
# axes = pca.components_
# XIItransformed = pca.transform(XIIcentered)
# XIImirrored = []
# for vertex in XIItransformed:
#     if vertex[1] > 0:
#         XIImirrored.append(vertex)
# 
# mirrored_center = np.mean(XIImirrored, axis=0)
# x, y, z = np.array(XIImirrored).T
# 
# fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, alphahull=0, color='lightpink', opacity=0.5)])
# #fig.add_trace(go.Scatter3d(x=[mirrored_center[0]], y=[mirrored_center[1]], z=[mirrored_center[2]], mode='markers', marker=dict(size=10, color='black')))
# 
# 
# pc1x = [0, axes[0][0]]
# pc1y = [0, axes[0][1]]
# pc1z = [0, axes[0][2]]
# pc2x = [0, axes[1][0]]
# pc2y = [0, axes[1][1]]
# pc2z = [0, axes[1][2]]
# pc3x = [0, axes[2][0]]
# pc3y = [0, axes[2][1]]
# pc3z = [0, axes[2][2]]
# 
# fig.add_trace(go.Scatter3d(x=pc1x, y=pc1y, z=pc1z, mode='lines', line=dict(color='black', width=4)))
# fig.add_trace(go.Scatter3d(x=pc2x, y=pc2y, z=pc2z, mode='lines', line=dict(color='black', width=4)))
# fig.add_trace(go.Scatter3d(x=pc3x, y=pc3y, z=pc3z, mode='lines', line=dict(color='black', width=4)))
# =============================================================================


fig.show()

# =============================================================================
# mirrored = []
# for vertex in XIIvertices:
#     if vertex[2] > 5700:
#         mirrored.append(vertex)
# 
# XIImesh.vertices = mirrored
# XIIsmoothed = XIImesh.clone().smooth()
#     
# 
# show(XIIsmoothed)
# =============================================================================

# =============================================================================
# pc1x = [mirrored_center[0], axes[0][0]]
# pc1y = [mirrored_center[1], axes[0][1]]
# pc1z = [mirrored_center[2], axes[0][2]]
# pc2x = [mirrored_center[0], axes[1][0]]
# pc2y = [mirrored_center[1], axes[1][1]]
# pc2z = [mirrored_center[2], axes[1][2]]
# pc3x = [mirrored_center[0], axes[2][0]]
# pc3y = [mirrored_center[1], axes[2][1]]
# pc3z = [mirrored_center[2], axes[2][2]]
# =============================================================================


# =============================================================================
# from scipy.spatial import Delaunay
# print('starting Delaunay transform')
# XIIforDelaunay = XIItransformed[:, :2]
# tris = Delaunay(XIIforDelaunay)
# print('Delaunay transform finished, creating transformed XII mesh')
# XIItransMesh = Mesh([XIItransformed, tris.simplices.T])
# XIIsmoothed = XIItransMesh.clone().smooth(niter=100)
# #XIIMeshsliced = XIIsmoothed.slice(origin=XIIorigin, normal=(0,0,-1))
# print('rendering XII mesh')
# =============================================================================
#show(XIIsmoothed, axes=1)


# =============================================================================
# XIIsystem = MeshCoordinateSystem(XIIvertices)
# XIIvertices = XIIsystem.local_vertices
# XIIvertices_global = XIIsystem.transform_to_global(XIIvertices)
# 
# XIIap = [point[0] for point in XIIvertices_global]
# XIIml = [point[1] for point in XIIvertices_global]
# XIIdv = [point[2] for point in XIIvertices_global]
# 
# pc1 = Line([XIIaxes_global[0], XIIaxes_global[0]+500])
# pc2 = Line([XIIaxes_global[1], XIIaxes_global[1]+450])
# pc3 = Line([XIIaxes_global[2], XIIaxes_global[2]+200])
# 
# ccf_scene.add(pc1)
# ccf_scene.add(pc2)
# ccf_scene.add(pc3)
# 
# ccf_scene.render(camera=cameras.sagcam)
# =============================================================================
# =============================================================================
# 
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# 
# def visualize_mesh_with_pca_axes(vertices, coord_system, axis_length=None):
#     """
#     Visualize mesh vertices with PCA axes overlaid.
#     
#     Parameters:
#     -----------
#     vertices : array-like, shape (n, 3)
#         Mesh vertices
#     coord_system : MeshCoordinateSystem
#         Your coordinate system object
#     axis_length : float, optional
#         Length of axis arrows. If None, auto-scales to mesh size
#     """
#     fig = plt.figure(figsize=(15, 5))
#     
#     # Auto-scale arrows if not specified
#     if axis_length is None:
#         mesh_span = np.ptp(vertices, axis=0).max()  # Range of mesh
#         axis_length = mesh_span * 0.4
#     
#     # --- Plot 1: Original mesh with PCA axes ---
#     ax1 = fig.add_subplot(131, projection='3d')
#     
#     # Plot vertices
#     ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
#                c='lightblue', alpha=0.3, s=1)
#     
#     # Plot origin
#     origin = coord_system.origin
#     ax1.scatter(*origin, c='black', s=100, marker='o', label='Origin')
#     
#     # Plot PCA axes as arrows
#     colors = ['red', 'green', 'blue']
#     labels = ['PC1 (AP)', 'PC2 (ML)', 'PC3 (DV)']
#     
#     for i, (color, label) in enumerate(zip(colors, labels)):
#         axis_vector = coord_system.axes[i] * axis_length
#         ax1.quiver(origin[0], origin[1], origin[2],
#                   axis_vector[0], axis_vector[1], axis_vector[2],
#                   color=color, arrow_length_ratio=0.3, linewidth=3,
#                   label=label)
#     
#     ax1.set_xlabel('X (global, μm)')
#     ax1.set_ylabel('Y (global, μm)')
#     ax1.set_zlabel('Z (global, μm)')
#     ax1.set_title('Mesh in Global Coordinates\nwith PCA Axes')
#     ax1.legend()
#     
#     # Make axes equal
#     set_axes_equal(ax1)
#     
#     # --- Plot 2: Transformed mesh (in PCA coordinates) ---
#     ax2 = fig.add_subplot(132, projection='3d')
#     
#     local_verts = coord_system.local_vertices
#     ax2.scatter(local_verts[:, 0], local_verts[:, 1], local_verts[:, 2],
#                c='lightblue', alpha=0.3, s=1)
#     
#     # Plot axes (now they're just the standard x, y, z axes)
#     ax2.quiver(0, 0, 0, axis_length, 0, 0,
#               color='red', arrow_length_ratio=0.3, linewidth=3,
#               label='PC1 (AP)')
#     ax2.quiver(0, 0, 0, 0, axis_length, 0,
#               color='green', arrow_length_ratio=0.3, linewidth=3,
#               label='PC2 (ML)')
#     ax2.quiver(0, 0, 0, 0, 0, axis_length,
#               color='blue', arrow_length_ratio=0.3, linewidth=3,
#               label='PC3 (DV)')
#     
#     ax2.set_xlabel('PC1: Anterior-Posterior (μm)')
#     ax2.set_ylabel('PC2: Medial-Lateral (μm)')
#     ax2.set_zlabel('PC3: Dorsal-Ventral (μm)')
#     ax2.set_title('Mesh in PCA Coordinates\n(Aligned)')
#     ax2.legend()
#     
#     set_axes_equal(ax2)
#     
#     # --- Plot 3: Variance explained ---
#     ax3 = fig.add_subplot(133)
#     
#     variance_ratio = coord_system.explained_variance / coord_system.explained_variance.sum()
#     bars = ax3.bar(['PC1\n(AP)', 'PC2\n(ML)', 'PC3\n(DV)'],
#                    variance_ratio * 100,
#                    color=['red', 'green', 'blue'], alpha=0.7)
#     
#     # Add percentage labels on bars
#     for bar, ratio in zip(bars, variance_ratio):
#         height = bar.get_height()
#         ax3.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{ratio*100:.1f}%',
#                 ha='center', va='bottom')
#     
#     ax3.set_ylabel('Variance Explained (%)')
#     ax3.set_title('Variance Along Each PC')
#     ax3.set_ylim(0, 100)
#     
#     plt.tight_layout()
#     return fig
# 
# def set_axes_equal(ax):
#     """Make axes of 3D plot have equal scale"""
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()
#     
#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)
#     
#     plot_radius = 0.5*max([x_range, y_range, z_range])
#     
#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
# 
# # Usage:
# ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True)
# ccf_scene.add_brain_region('XII', color='orange', alpha=0.1, silhouette=False)
# medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
# ccf_scene.slice(plane=medplane)
# actors = ccf_scene.get_actors()
# root = actors[0]
# root._needs_silhouette = False
# XIImesh = actors[1]
# XIIvertices = XIImesh.mesh.vertices
# analyzer = MeshCoordinateSystem(XIIvertices)
# fig = visualize_mesh_with_pca_axes(XIIvertices, analyzer)
# plt.show()
# =============================================================================

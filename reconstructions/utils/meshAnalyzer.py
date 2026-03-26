# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:14:06 2026
s/o claude thru terriergpt, writing an implementation to transform coordinates into mesh space
@author: economolab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


class MeshCoordinateSystem:
    def __init__(self, vertices):
        """
        Define a coordinate system based on mesh vertices.
        
        Parameters:
        -----------
        vertices : array-like, shape (n, 3)
            The vertices of the mesh
        """
        self.vertices = np.array(vertices)
        
        # Center the mesh
        self.origin = np.mean(self.vertices, axis=0)
        centered_vertices = self.vertices - self.origin
        
        # Use PCA to find principal axes
        pca = PCA(n_components=3)
        pca.fit(centered_vertices)
        
        # Principal components become our new axes
        # PC1 (longest axis) = anterior-posterior
        # PC2 (second longest) = medial-lateral  
        # PC3 (shortest) = dorsal-ventral
        self.axes = pca.components_  # Shape (3, 3)
        self.explained_variance = pca.explained_variance_
        
        # Transform vertices to local coordinates
        self.local_vertices = self.transform_to_local(self.vertices)
        
    def transform_to_local(self, points):
        """
        Transform points from global to mesh-local coordinates.
        
        Parameters:
        -----------
        points : array-like, shape (n, 3)
            Points in global coordinates
            
        Returns:
        --------
        local_points : ndarray, shape (n, 3)
            Points in mesh-local coordinates
        """
        points = np.array(points)
        centered = points - self.origin
        # Project onto new axes
        local = centered @ self.axes.T
        return local
    
    def transform_to_global(self, local_points):
        """
        Transform points from mesh-local to global coordinates.
        """
        local_points = np.array(local_points)
        # Inverse transformation
        global_points = local_points @ self.axes + self.origin
        return global_points


class MeshDensityAnalyzer:
    def __init__(self, mesh_vertices):
        """
        Analyze point density within a mesh, normalized by cross-sectional area.
        
        Parameters:
        -----------
        mesh_vertices : array-like, shape (n, 3)
            Vertices of the mesh
        mesh_faces : array-like, shape (m, 3), optional
            Faces of the mesh (for better area calculation)
        """
        self.coord_system = MeshCoordinateSystem(mesh_vertices)

            
    def compute_cross_sectional_areas(self, axis=0, bin_size=10):
        """
        Compute cross-sectional area of mesh at different positions along an axis.
        
        Parameters:
        -----------
        axis : int (0, 1, or 2)
            Which axis to slice along (0=AP, 1=ML, 2=DV)
        bin_size : float
            Size of bins in microns
            
        Returns:
        --------
        bin_centers : ndarray
            Center position of each bin
        areas : ndarray
            Cross-sectional area at each bin
        """
        local_verts = self.coord_system.local_vertices
        
        # Define bins along the axis
        axis_min = local_verts[:, axis].min()
        axis_max = local_verts[:, axis].max()
        bins = np.arange(axis_min, axis_max + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        areas = []
        
        for center in bin_centers:
            # Find vertices within a slice
            mask = np.abs(local_verts[:, axis] - center) < bin_size/2
            slice_verts = local_verts[mask]
            
            if len(slice_verts) < 3:
                areas.append(0)
                continue
            
            # Project to 2D (remove the slicing axis)
            other_axes = [i for i in range(3) if i != axis]
            verts_2d = slice_verts[:, other_axes]
            
            try:
                # Compute convex hull area of the slice
                hull = ConvexHull(verts_2d)
                areas.append(hull.volume)  # In 2D, 'volume' is area
            except:
                areas.append(0)
        
        return bin_centers, np.array(areas)
    
    def analyze_point_density(self, points, axis=0, bin_size=10):
        """
        Analyze density of points within mesh, normalized by cross-sectional area.
        
        Parameters:
        -----------
        points : array-like, shape (n, 3)
            Points to analyze (in global coordinates)
        axis : int
            Axis to bin along (0=AP, 1=ML, 2=DV)
        bin_size : float
            Bin size in microns
            
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'bin_centers': position of each bin
            - 'counts': number of points in each bin
            - 'areas': cross-sectional area of each bin
            - 'density': points per unit area (counts/areas)
            - 'local_points': transformed point coordinates
        """
        # Transform points to local coordinates
        local_points = self.coord_system.transform_to_local(points)
        
        # Get cross-sectional areas
        bin_centers, areas = self.compute_cross_sectional_areas(axis, bin_size)
        
        # Bin the points
        axis_coords = local_points[:, axis]
        axis_min = bin_centers[0] - bin_size/2
        axis_max = bin_centers[-1] + bin_size/2
        
        counts, bin_edges = np.histogram(axis_coords, 
                                        bins=len(bin_centers),
                                        range=(axis_min, axis_max))
        
        # Calculate density (avoid division by zero)
        density = np.zeros_like(counts, dtype=float)
        nonzero_mask = areas > 0
        density[nonzero_mask] = counts[nonzero_mask] / areas[nonzero_mask]
        
        return {
            'bin_centers': bin_centers,
            'counts': counts,
            'areas': areas,
            'density': density,
            'local_points': local_points
        }
    
    def plot_density_profile(self, results, axis=0, normalize=True):
        """
        Plot density profile along an axis.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_point_density
        axis : int
            Which axis was analyzed
        normalize : bool
            If True, plot density; if False, plot raw counts
        """
        axis_names = ['Anterior-Posterior', 'Medial-Lateral', 'Dorsal-Ventral']
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot density or counts
        if normalize:
            axes[0].bar(results['bin_centers'], results['density'], 
                       width=results['bin_centers'][1]-results['bin_centers'][0],
                       alpha=0.7)
            axes[0].set_ylabel('Density (points/μm²)', fontsize=12)
            axes[0].set_title(f'Point Density along {axis_names[axis]} Axis')
        else:
            axes[0].bar(results['bin_centers'], results['counts'],
                       width=results['bin_centers'][1]-results['bin_centers'][0],
                       alpha=0.7)
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_title(f'Point Count along {axis_names[axis]} Axis')
        
        # Plot cross-sectional area
        axes[1].plot(results['bin_centers'], results['areas'], 'r-', linewidth=2)
        axes[1].fill_between(results['bin_centers'], results['areas'], alpha=0.3)
        axes[1].set_ylabel('Cross-sectional Area (μm²)', fontsize=12)
        axes[1].set_xlabel(f'{axis_names[axis]} Position (μm)', fontsize=12)
        axes[1].set_title('Mesh Cross-sectional Area')
        
        plt.tight_layout()
        return fig



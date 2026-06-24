# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:33:43 2026

@author: samkr
"""

from brainrender import Scene

ccf_scene = Scene(atlas_name="allen_mouse_10um")
root = ccf_scene.get_actors()[0]
root._needs_silhouette=False

ccf_scene.add_brain_region('ARH', silhouette=False, color='red', alpha=0.5)

ccf_scene.render()
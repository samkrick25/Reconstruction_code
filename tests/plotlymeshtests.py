# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:33:37 2026

@author: economolab
"""

import plotly.graph_objects as go
import numpy as np

# Download data set from plotly repo
pts = np.loadtxt(np.lib.npyio.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
x, y, z = pts.T

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
fig.show()
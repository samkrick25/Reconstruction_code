# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:10:56 2026

@author: samkr
"""

import numpy as np

x = 13326.9
x = x/10

print(np.round(x).astype(np.uint16))

new = tuple([7, 5, 9])
node = {'x': 133, 'y': 777, 'z': 000}
node['x'], node['y'], node['z'] = new
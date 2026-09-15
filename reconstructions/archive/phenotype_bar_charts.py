# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:16:29 2026

@author: samkr
"""
import matplotlib.pyplot as plt

vals = [4, 36, 23, 22, 6, 26, 3]
labs = ['forebrain', 'GRN', 'mossy', 'premotor', 'sensorimotor', 'sensory', 'vestibular']

fig, ax = plt.subplots()
fig.suptitle('Distribution of phenotypes in IRN/PARN')
ax.bar(labs, vals)
ax.tick_params(rotation=45, axis='x')

#%%
fig, ax = plt.subplots()
vals = [0,2,3,2,0,2,0]
fig.suptitle('Distribution of phenotypes in 708369')
ax.bar(labs, vals)
ax.tick_params(rotation=45, axis='x')

#%%
fig, ax = plt.subplots()
vals = [0,1,7,0,0,2,1]
fig.suptitle('Distribution of phenotypes in Pvalb')
ax.bar(labs, vals)
ax.tick_params(rotation=45, axis='x')

#%%
fig, ax = plt.subplots()
vals = [0,9,0,2,1,9,0]
fig.suptitle('Distribution of phenotypes in Vgat-cre')
ax.bar(labs, vals)
ax.tick_params(rotation=45, axis='x')

#%%
fig, ax = plt.subplots()
vals = [0,2,5,5,0,1,0]
fig.suptitle('Distribution of phenotypes in AiE2255')
ax.bar(labs, vals)
ax.tick_params(rotation=45, axis='x')

#%%
fig, ax = plt.subplots()
vals = [0,6,3,7,0,5,1]
fig.suptitle('Distribution of phenotypes receiving ALM input')
ax.bar(labs, vals)
ax.tick_params(rotation=45, axis='x')

# %%
fig, ax = plt.subplots()
vals = [9,11,21,1,3,4,4,4,5,3,7,13,22]
labs = ['AiE2256', 'Pvalb', 'Vgat', 'Calb2', 'Cart', 'Crh', 'Dbh', 'Gal', 'Grp', 'Sim1', 'Som', 'AiE2255', 'ALM input']
ax.bar(labs, vals)
fig.suptitle('Distribution of cells in genotypes')
ax.tick_params(rotation=45, axis='x')
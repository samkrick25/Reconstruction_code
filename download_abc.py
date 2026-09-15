# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:08:20 2026
analysis of 251 nts dbh glut type
@author: samkr
"""

from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache as apc
import os

projDir = r"D:\allen_brain_atlas"
abc_cache = apc.from_cache_dir(projDir)

c_man = abc_cache.current_manifest

abc_cache.load_manifest(c_man)


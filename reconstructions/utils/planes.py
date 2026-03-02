from brainrender import Scene

ccf_scene = Scene(atlas_name='allen_mouse_10um')

allplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,0))
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
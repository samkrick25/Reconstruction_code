from brainrender import Scene
from reconstructions.utils.cameras import corcam, topcam, sagcam

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=False) #, title='PT upper endpoint distribution in GPe and STR'
# ccf_scene.add_brain_region('STR', color='red', alpha=0.05, silhouette=False)
# ccf_scene.add_brain_region('GPe', color='blue', alpha=0.05, silhouette=False)
#ccf_scene.add_brain_region('PG', color='orange', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('CL', color='red', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('PF', color='blue', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('MD', color='cyan', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('LP', color='yellow', alpha=0.2, silhouette=False)
ccf_scene.add_brain_region('PO', color='orange', alpha=0.1, silhouette=False)

#slice scene to show ROI
#antplanebg=ccf_scene.atlas.get_plane(pos=(3250,4000,5000),plane='frontal')
#posplanebg=ccf_scene.atlas.get_plane(pos=(7750,4000,5000),norm=(-1, 0, 0),plane='frontal')
#antplanepg=ccf_scene.atlas.get_plane(pos=(8500,4000,5000),plane='frontal')
#posplanepg=ccf_scene.atlas.get_plane(pos=(9500,4000,5000),norm=(-1,0,0),plane='frontal')
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
#allplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,0))
#ccf_scene.slice(plane=antplanebg)
#ccf_scene.slice(plane=posplanebg)
# actors = ccf_scene.get_actors() #just debug and look here to find actor indices
# root_ccf = actors[0]
ccf_scene.slice(plane=medplane)
#ccf_scene.slice(plane=allplane) #, actors=root_ccf
ccf_scene.render(camera=topcam)

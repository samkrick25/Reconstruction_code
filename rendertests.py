from brainrender import Scene
from brainrender.actors import Neuron
import pandas as pd
import numpy as np
from vedo import Lines

cellpath = r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_noAA\swcN016-715345-HD.swc"

# ----------------------------------------------------------------
# Step 1: Verify morphio isn't dropping nodes either
# ----------------------------------------------------------------
swc = pd.read_csv(
    cellpath,
    comment='#',
    sep=r'\s+',
    names=['id', 'type', 'x', 'y', 'z', 'r', 'parent']
)

try:
    from morphio import SectionType
    from morphapi.morphology.morphology import Neuron as MorphNeuron

    morph     = MorphNeuron(data_file=cellpath)
    morpho    = morph.morphology

    morphio_axon = sum(
        len(s.points) for s in morpho.sections
        if s.type == SectionType.axon
    )
    morphio_dend = sum(
        len(s.points) for s in morpho.sections
        if s.type == SectionType.basal_dendrite
    )
    swc_axon = len(swc[swc['type'] == 2])
    swc_dend = len(swc[swc['type'] == 3])

    print("=== morphio vs SWC node counts ===")
    print(f"Axon     — SWC: {swc_axon:>6}   morphio: {morphio_axon:>6}   "
          f"{'OK' if morphio_axon == swc_axon else 'MISMATCH <---'}")
    print(f"Dendrite — SWC: {swc_dend:>6}   morphio: {morphio_dend:>6}   "
          f"{'OK' if morphio_dend == swc_dend else 'MISMATCH <---'}")

except Exception as e:
    print(f"morphio check failed: {e}")


# ----------------------------------------------------------------
# Step 2: Build line actors directly from the SWC, bypassing
# tube mesh generation entirely
# ----------------------------------------------------------------
def swc_to_line_actors(swc_df, axon_color='blue', dendrite_color='red', lw=2):
    """
    Build vedo Lines actors for axon and dendrite segments directly
    from a parsed SWC dataframe. Bypasses morphapi/vedo tube merging.
    """
    node_coords = swc_df.set_index('id')[['x', 'y', 'z']]

    # Only rows that have a valid parent
    has_parent = swc_df[swc_df['parent'] != -1]

    actors = []
    for neurite_type, color in [(2, axon_color), (3, dendrite_color)]:
        subset = has_parent[has_parent['type'] == neurite_type]
        if subset.empty:
            continue

        # Build (N, 3) start and end point arrays
        start_pts = node_coords.loc[subset['id']].values
        end_pts   = node_coords.loc[subset['parent']].values

        actor = Lines(start_pts, end_pts, c=color, lw=lw)
        actors.append(actor)

    return actors


# ----------------------------------------------------------------
# Step 3: Render in brainrender scene
# ----------------------------------------------------------------
ccf_scene = Scene(atlas_name='allen_mouse_10um')

line_actors = swc_to_line_actors(swc, axon_color='blue', dendrite_color='red', lw=2)
for actor in line_actors:
    ccf_scene.add(actor)

ccf_scene.render()
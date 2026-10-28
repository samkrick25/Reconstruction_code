# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:24:26 2026
write swc from json, written by terriergpt claude 4.6 sonnet
@author: economolab
"""

import json
import os
import glob


def merge_nodes(axon_nodes, dendrite_nodes, id_string='?'):
    if not axon_nodes and not dendrite_nodes:
        return []

    # Safe soma lookup
    axon_soma = next(
        (n for n in axon_nodes if n['structureIdentifier'] == 1), None
    )
    if axon_soma is None:
        print(f"  WARNING [{id_string}]: No soma found in axon — "
              f"falling back to first axon node.")
        axon_soma = axon_nodes[0]
    axon_soma_id = axon_soma['sampleNumber']

    # ----------------------------------------------------------------
    # Fix orphaned roots in the axon array
    # (fragments with parentNumber==-1 that are not the soma)
    # ----------------------------------------------------------------
    fixed_axon = []
    for node in axon_nodes:
        new_node = dict(node)
        if (node['parentNumber'] == -1
                and node['sampleNumber'] != axon_soma_id):
            print(f"  INFO [{id_string}]: Axon fragment root at "
                  f"sampleNumber {node['sampleNumber']} — "
                  f"connecting to soma.")
            new_node['parentNumber'] = axon_soma_id
        fixed_axon.append(new_node)

    if not dendrite_nodes:
        return fixed_axon

    # ----------------------------------------------------------------
    # Renumber and merge dendrite nodes
    # ----------------------------------------------------------------
    axon_ids = {n['sampleNumber'] for n in fixed_axon}
    offset = max(axon_ids)

    dendrite_soma_ids = {
        n['sampleNumber'] for n in dendrite_nodes if n['structureIdentifier'] == 1
    }

    non_soma_nodes = [n for n in dendrite_nodes if n['structureIdentifier'] != 1]
    non_soma_ids   = {n['sampleNumber'] for n in non_soma_nodes}
    id_map = {n['sampleNumber']: n['sampleNumber'] + offset
              for n in non_soma_nodes}

    renumbered = []
    for node in non_soma_nodes:
        new_node = dict(node)
        new_node['sampleNumber'] = id_map[node['sampleNumber']]

        parent = node['parentNumber']
        if parent in non_soma_ids:
            # Points to another dendrite node — remap
            new_node['parentNumber'] = id_map[parent]
        elif parent in dendrite_soma_ids:
            # Points to duplicate dendrite soma — re-parent to real soma
            new_node['parentNumber'] = axon_soma_id
        elif parent == -1:
            # Orphaned dendrite fragment root — connect to soma
            print(f"  INFO [{id_string}]: Dendrite fragment root at "
                  f"sampleNumber {node['sampleNumber']} — "
                  f"connecting to soma.")
            new_node['parentNumber'] = axon_soma_id
        # else: parent already points into axon array — leave unchanged

        renumbered.append(new_node)

    return fixed_axon + renumbered


def json_to_swc(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"Found {len(json_files)} JSON files.")

    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        neurons = data.get('neurons', [])
        if not neurons:
            print(f"  WARNING: No neurons found in {json_path}, skipping.")
            continue

        for neuron in neurons:
            id_string = neuron.get('idString', 'unknown')

            axon_nodes     = neuron.get('axon', [])
            dendrite_nodes = neuron.get('dendrite', [])

            all_nodes = merge_nodes(axon_nodes, dendrite_nodes)

            if not all_nodes:
                print(f"  WARNING: No nodes found for {id_string}, skipping.")
                continue

            all_nodes.sort(key=lambda n: n['sampleNumber'])

            swc_path = os.path.join(output_dir, f'{id_string}.swc')

            with open(swc_path, 'w') as f:
                f.write(f'# {id_string}\n')
                doi = neuron.get('DOI', '')
                if doi:
                    f.write(f'# DOI: {doi}\n')
                f.write('# nodeID structureIdentifier x y z radius parent\n')

                for node in all_nodes:
                    f.write(
                        f"{node['sampleNumber']} "
                        f"{node['structureIdentifier']} "
                        f"{node['x']} "
                        f"{node['y']} "
                        f"{node['z']} "
                        f"{node['radius']} "
                        f"{node['parentNumber']}\n"
                    )

            print(f"  Written: {swc_path} "
                  f"({len(axon_nodes)} axon + "
                  f"{len(dendrite_nodes)} dendrite nodes, "
                  f"{len(all_nodes)} total written)")


if __name__ == '__main__':
    input_dir  = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\json_coordswapped'
    output_dir = r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson"
    json_to_swc(input_dir, output_dir)
            
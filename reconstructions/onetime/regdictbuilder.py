from reconstructions.utils import load_data
from reconstructions.utils.filedirs import allcoordswapped
import json
import os

_, _, aidtoreg, _, _ = load_data.load_neurons(allcoordswapped)

filename = 'aidtoreg.json'
savefolder = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data'
savepath = os.path.join(savefolder, filename)
try:
    with open(savepath, 'w') as f:
        json.dump(aidtoreg, f)
except IOError as e:
    print(f'Error saving file {e}')
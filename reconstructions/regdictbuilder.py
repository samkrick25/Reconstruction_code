from utils import load_data
from utils.filedirs import alldir
import json

_, _, aidtoreg, _, _ = load_data.load_neurons(alldir)

filename = 'aidtoreg.json'
try:
    with open(filename, 'w') as f:
        json.dump(aidtoreg, f)
except IOError as e:
    print(f'Error saving file {e}')
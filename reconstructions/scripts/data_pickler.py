from reconstructions.utils import load_data
import pickle
from reconstructions.utils.filedirs import allcoordswapped

cells, somas, _, _, _ = load_data.load_neurons(allcoordswapped)
# freqs = load_data.get_frequencies(cells, somas)

# savefile = r'reconstructions\data\freqs.pkl'
# pickle.dump(freqs, open(savefile, 'wb'))

savefilesomas = r'reconstructions\data\somas.pkl'
pickle.dump(somas, open(savefilesomas, 'wb'))
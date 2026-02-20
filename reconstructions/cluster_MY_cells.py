from utils import load_data, preprocess_funcs, metrics_funcs
from utils.filedirs import alldir
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MIDLINEZ = 5750

cells, somas, _, _, _ = load_data.load_neurons(alldir)

frequencies = load_data.get_frequencies(cells, somas)

ppfreqs = preprocess_funcs.preprocess(frequencies)


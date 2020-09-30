from torch import from_numpy
import pickle as pkl
with open('weights1.pkl', 'rb') as wp:
    name_weights = pkl.load(wp)
    state_dict = {}
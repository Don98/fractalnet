from torch import from_numpy
import pickle as pkl
with open('weights1.pkl', 'rb') as wp:
    name_weights = pkl.load(wp)
    state_dict = {}
    print(name_weights['concat_pool3_7_plus'])
    #print(name_weights["conv2_12"]["weight"].shape)
    exit()
    for i in name_weights.keys():
        print(i)
        print("-"*50)
        for j in name_weights[i].keys():
            print(j)
        print("="*50)

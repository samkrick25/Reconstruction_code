import json
import os

#likely both one-time use scripts here, might have to be done when new neurons are acquired

#to rename idstring in json 

# folder1 = r"C:\Users\samkr\OneDrive\Documents\BU\AllenExASPIM\medulla_IRN_PRN_PGRN\json\json"
# folder2 = r"C:\Users\samkr\OneDrive\Documents\BU\AllenExASPIM\medulla_IRN_PRN_PGRN\json_w_names"
# for filename in os.listdir(folder1):
#     file  = os.path.join(folder1, filename)
#     with open(file, 'r') as f:
#         data = json.load(f)
#         #print(type(data['neurons']))
#         neuron = data['neurons']
#         meta = neuron[0]
#         #print(type(meta))
#         meta['idString'] = filename.replace('.json','')
#         #print(meta['idString'])
#         new_name = meta['idString'] + '.json'
#         full_name = os.path.join(folder2, filename)
#         with open(full_name, 'w') as f2:
#             json.dump(data, f2)

#change 'neuron' in AA cells to 'neurons' to match the rest of the data 
folder1 = r"C:\Users\samkr\OneDrive\Documents\BU\AllenExASPIM\medulla_IRN_PRN_PGRN\AA_og"
folder2 = r"C:\Users\samkr\OneDrive\Documents\BU\AllenExASPIM\medulla_IRN_PRN_PGRN\AA-rename"
for filename in os.listdir(folder1):
    file = os.path.join(folder1, filename)
    with open(file,'r') as f:
        data = json.load(f)
        data['neurons'] = data.pop('neuron')
        full_name = os.path.join(folder2, filename)
        with open(full_name,'w') as f2:
            json.dump(data, f2)
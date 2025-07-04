import os

folder = r'C:\Users\samkr\OneDrive\Documents\BU\AllenExASPIM\medulla_IRN_PRN_PGRN\json'

unique = []
for file in os.listdir(folder):
    parts = file.split("-")
    #print(parts)
    if parts[1] not in unique:
        unique.append(parts[1])
print(len(unique), unique)
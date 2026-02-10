from utils import load_data

dir = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\IRN'

_, somas, aidtoreg, _, _ = load_data.load_neurons(dir)
somacomps = {}
for key, value in somas.items():
    somacomps[key] = aidtoreg[somas[key]['allenId']]

IRNcells = [key for key, value in somacomps.items() if value[1] == 'IRN']
PARNcells = [key for key, value in somacomps.items() if value[1] == 'PARN']
othercells = []
for key, value in somacomps.items():
    if key not in IRNcells and key not in PARNcells:
        othercells.append(key)
    else:
        continue


print(IRNcells, len(IRNcells))
print()
print(PARNcells, len(PARNcells))
print()
print(othercells)

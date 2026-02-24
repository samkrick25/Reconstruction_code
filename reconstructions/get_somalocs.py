from utils import load_data

dir = r"C:\Data\reconstructions\for_bu\all_json"

_, somas, aidtoreg, _, _ = load_data.load_neurons(dir)
somacomps = {}
for key, value in somas.items():
    try:
        somacomps[key] = aidtoreg[somas[key]['allenId']]
    except KeyError:
        print(f'{key} not found')

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
print(othercells, len(othercells))

import json

d = json.load(open('input.json'))[:1]
sol = d[0]['solutions'][0]
pps = sol.get('prm_processed_solution', '')
print('ки count:', pps.count(' ки'))
print('sample:', pps[:800])
print('\n--- all keys in solution ---')
print(sol.keys())

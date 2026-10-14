import json, time, sys
import numpy as np
from sim import *

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 250
VAL = list(range(3000, 3006))
agent = QAgent(alpha=0.2, eps=1.0, seed=42)
log = []
t0 = time.time()
for k in range(N_EP):
    agent.eps = max(0.05, 0.97 ** k)
    agent.alpha = max(0.05, 0.2 * 0.985 ** k)
    seed = 1000 + k
    s = Sim(seed=seed, **day_config(seed))
    r = s.run(None, learn=True, agent=agent)
    row = dict(ep=k, eps=agent.eps, alpha=agent.alpha, train_reward=r['reward'], train_wait=r['avg_wait'],
               train_rides=r['rides'], train_fuel=r['fuel_litres'])
    if k % 5 == 0 or k == N_EP - 1:
        pol = QPolicy(agent.Q, agent.N, min_visits=1)
        vr = [Sim(seed=v, **day_config(v)).run(pol) for v in VAL]
        row.update(val_reward=float(np.mean([x['reward'] for x in vr])), val_wait=float(np.mean([x['avg_wait'] for x in vr])),
                   val_rides=float(np.mean([x['rides'] for x in vr])), val_fuel=float(np.mean([x['fuel_litres'] for x in vr])))
    log.append(row)
    if k % 10 == 0: print(k, round(time.time() - t0), {a: round(b, 2) for a, b in row.items()}, flush=True)
Q, N = agent.arrays()
np.save('q_table.npy', Q); np.save('visits.npy', N)
json.dump(log, open('train_log.json', 'w'))
json.dump(agent.sample_update, open('sample_update.json', 'w'))
print('done', round(time.time() - t0), 'sec; states visited:', int((N.sum(1) > 0).sum()), 'of', N_STATES)

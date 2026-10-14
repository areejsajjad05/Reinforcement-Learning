import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
res = json.load(open('results.json'))
COL = {'Random action': '#9aa0a6', 'Send to busiest zone': '#d9534f', 'Stay put (status quo)': '#f0ad4e', 'Rule-based heuristic': '#5bc0de', 'Q-learning agent': '#0b6b3a'}
names = list(res['policies'].keys())
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, (k, t, sc) in zip(ax, [('avg_wait', 'Avg passenger wait (min)', 1), ('rides', 'Rides completed / day', 1), ('fuel_litres', 'Fuel used / day (litres)', 1), ('cancel_rate', 'Cancelled requests (%)', 100)]):
    m = np.array([res['policies'][n][k]['mean'] * sc for n in names]); sd = np.array([res['policies'][n][k]['std'] * sc for n in names])
    a.bar(range(len(names)), m, yerr=[np.minimum(sd, m), sd], color=[COL[n] for n in names], capsize=2)
    a.set_xticks(range(len(names))); a.set_xticklabels(['Random', 'Busiest', 'Stay', 'Rule', 'Q-learn'], fontsize=8); a.set_title(t, fontsize=9)
    if k == 'avg_wait': a.axhline(7, color='k', ls=':', lw=1); a.text(-0.4, 7.15, 'target 7 min', fontsize=7)
    if k == 'rides': a.set_ylim(10000, 11900)
plt.tight_layout(); plt.savefig('fig_metrics.png'); plt.close()

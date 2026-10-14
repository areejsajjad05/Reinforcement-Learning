import json, numpy as np
from sim import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
log = json.load(open('train_log.json')); res = json.load(open('results.json'))
base = [Sim(seed=1000 + r['ep'], **day_config(1000 + r['ep'])).run(StayPolicy())['reward'] for r in log]
gain = np.array([(r['train_reward'] / b - 1) * 100 for r, b in zip(log, base)])
ep = [r['ep'] for r in log]; G = '#0b6b3a'
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].axhline(0, color='#f0ad4e', ls='--', lw=1.3, label='Stay put (status quo) = 0%')
ax[0].plot(ep, gain, color='#b7d7c4', lw=0.8, label='Single training day (exploring)')
ax[0].plot(ep[9:], np.convolve(gain, np.ones(10) / 10, mode='valid'), color=G, lw=2, label='10-episode moving average')
ax[0].set_xlabel('Training episode (1 episode = 1 simulated day)'); ax[0].set_ylabel('Reward vs status quo, same day (%)')
ax[0].set_title('Training reward curve (relative to baseline)'); ax[0].legend(frameon=False, fontsize=8, loc='lower right')
vx = [r['ep'] for r in log if 'val_reward' in r]; vy = np.array([r['val_reward'] for r in log if 'val_reward' in r]) / 1e6
ax[1].plot(vx, vy, color=G, lw=2, marker='o', ms=3, label='Q-learning (greedy)')
COL = {'Random action': '#9aa0a6', 'Send to busiest zone': '#d9534f', 'Stay put (status quo)': '#f0ad4e', 'Rule-based heuristic': '#5bc0de'}
for n, c in res['val_baselines'].items(): ax[1].axhline(c / 1e6, color=COL[n], ls='--', lw=1.3, label=n)
ax[1].set_xlabel('Training episode'); ax[1].set_ylabel('Mean reward on 6 held-out days (Rs million)'); ax[1].set_title('Greedy policy vs baselines (validation)')
ax[1].legend(frameon=False, fontsize=7, loc='center right', bbox_to_anchor=(1.0, 0.45))
plt.tight_layout(); plt.savefig('fig_reward_curve.png'); plt.close()
print('first10 mean gain %.2f%%, last50 mean gain %.2f%%' % (gain[:10].mean(), gain[-50:].mean()))

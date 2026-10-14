import json, numpy as np
from sim import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = np.load('q_table.npy'); N = np.load('visits.npy')
Ql, Nl = Q.tolist(), N.tolist()
TEST = list(range(5000, 5030)); VAL = list(range(3000, 3006))
KEYS = ['reward','avg_wait','p90_wait','rides','cancel_rate','fuel_litres','empty_share','net_revenue','min_zone_rate','zones_below_70','outskirt_rate','central_rate']

def pols():
    return {'Random action': RandomPolicy(7), 'Send to busiest zone': BusiestPolicy(), 'Stay put (status quo)': StayPolicy(),
            'Rule-based heuristic': HeuristicPolicy(), 'Q-learning agent': QPolicy(Ql, Nl, min_visits=30)}

def run_set(pol, seeds, **kw):
    return [Sim(seed=s, **day_config(s)).run(pol, **kw) for s in seeds]

res = {'policies': {}}
raw = {}
for name, pol in pols().items():
    rs = run_set(pol, TEST); raw[name] = rs
    res['policies'][name] = {k: dict(mean=float(np.mean([r[k] for r in rs])), std=float(np.std([r[k] for r in rs]))) for k in KEYS}
    res['policies'][name]['zone_rate'] = np.mean([r['zone_rate'] for r in rs], axis=0).tolist()
    if name == 'Q-learning agent':
        res['fallback_share'] = pol.fallback_uses / max(1, pol.total_uses)

def paired(a, b, k='reward', n=4000):
    d = np.array([x[k] for x in raw[a]]) - np.array([x[k] for x in raw[b]])
    rng = np.random.RandomState(0); bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), wins=int((d > 0).sum()), n=len(d))
res['paired'] = {'Q vs stay': paired('Q-learning agent', 'Stay put (status quo)'),
                 'Q vs heuristic': paired('Q-learning agent', 'Rule-based heuristic'),
                 'Q vs busiest': paired('Q-learning agent', 'Send to busiest zone'),
                 'Q vs random': paired('Q-learning agent', 'Random action'),
                 'Q vs stay (wait)': paired('Q-learning agent', 'Stay put (status quo)', 'avg_wait')}

# ---- pilot on 10% of vehicles (30 cars), same days; breaker + fallback active
mask = [i < 30 for i in range(NV)]
pil = []; 
for s in TEST:
    q = QPolicy(Ql, Nl, min_visits=30)
    pil.append(Sim(seed=s, **day_config(s)).run(q, mask=mask, base=StayPolicy(), fallback=HeuristicPolicy(), breaker_wait=3.0, breaker_km=7.0))
res['pilot'] = dict(
    ctrl_on=dict(rides_per_veh=float(np.mean([r['ctrl_on']['rides'] / 30 for r in pil])), wait=float(np.mean([r['ctrl_on']['avg_wait'] for r in pil])),
                 net_per_veh=float(np.mean([r['ctrl_on']['net_revenue'] / 30 for r in pil])), km_empty_per_veh=float(np.mean([r['ctrl_on']['km_empty'] / 30 for r in pil]))),
    ctrl_off=dict(rides_per_veh=float(np.mean([r['ctrl_off']['rides'] / 270 for r in pil])), wait=float(np.mean([r['ctrl_off']['avg_wait'] for r in pil])),
                  net_per_veh=float(np.mean([r['ctrl_off']['net_revenue'] / 270 for r in pil])), km_empty_per_veh=float(np.mean([r['ctrl_off']['km_empty'] / 270 for r in pil]))),
    fleet=dict(avg_wait=float(np.mean([r['avg_wait'] for r in pil])), cancel_rate=float(np.mean([r['cancel_rate'] for r in pil])), tripped=int(sum(r['tripped'] for r in pil))))
pn = np.array([r['ctrl_on']['net_revenue'] / 30 - r['ctrl_off']['net_revenue'] / 270 for r in pil])
rng = np.random.RandomState(1); bs = [rng.choice(pn, len(pn)).mean() for _ in range(4000)]
res['pilot']['net_diff'] = dict(mean=float(pn.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

# ---- safety test: a broken agent (send-to-busiest) is piloted; compare breaker on / off
sf = {'no_breaker': [], 'breaker': [], 'clean_fleet': []}
for s in TEST[:12]:
    cfg = day_config(s)
    sf['no_breaker'].append(Sim(seed=s, **cfg).run(BusiestPolicy(), mask=mask, base=StayPolicy()))
    sf['breaker'].append(Sim(seed=s, **cfg).run(BusiestPolicy(), mask=mask, base=StayPolicy(), fallback=HeuristicPolicy(), breaker_km=7.0, breaker_wait=3.0))
    sf['clean_fleet'].append(Sim(seed=s, **cfg).run(StayPolicy()))
res['safety'] = {k: dict(pilot_net_per_veh=float(np.mean([r['ctrl_on']['net_revenue'] / (300 if k == 'clean_fleet' else 30) for r in v])),
                         pilot_empty_km_per_veh=float(np.mean([r['ctrl_on']['km_empty'] / (300 if k == 'clean_fleet' else 30) for r in v])),
                         fleet_wait=float(np.mean([r['avg_wait'] for r in v])), fleet_cancel=float(np.mean([r['cancel_rate'] for r in v])),
                         tripped=int(sum(r['tripped'] for r in v)), n=len(v),
                         trip_hour=float(np.mean([START_HOUR + (r['trip_step'] or 0) * 5 / 60 for r in v if r['tripped']])) if any(r['tripped'] for r in v) else None)
                 for k, v in sf.items()}

# ---- day-type breakdown
dt = {}
for name in ['Stay put (status quo)', 'Rule-based heuristic', 'Q-learning agent']:
    rows = raw[name]; out = {}
    for lab, f in [('dry weekday', lambda c: not c['rain'] and not c['friday']), ('rain', lambda c: c['rain']), ('friday', lambda c: c['friday'])]:
        idx = [j for j, sd in enumerate(TEST) if f(day_config(sd))]
        out[lab] = dict(n=len(idx), wait=float(np.mean([rows[j]['avg_wait'] for j in idx])), cancel=float(np.mean([rows[j]['cancel_rate'] for j in idx])))
    dt[name] = out
res['day_types'] = dt
# ---- baselines on validation days for the learning curve
valb = {}
for name, pol in pols().items():
    if name == 'Q-learning agent': continue
    valb[name] = float(np.mean([r['reward'] for r in run_set(pol, VAL)]))
res['val_baselines'] = valb

# ---- one detailed day for the app-style chart
res['state_count'] = dict(total=N_STATES, visited=int((N.sum(1) > 0).sum()), well_visited=int((N.sum(1) >= 30).sum()))
json.dump(res, open('results.json', 'w'), indent=1)

# ======================= figures
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
COL = {'Random action': '#9aa0a6', 'Send to busiest zone': '#d9534f', 'Stay put (status quo)': '#f0ad4e', 'Rule-based heuristic': '#5bc0de', 'Q-learning agent': '#0b6b3a'}
log = json.load(open('train_log.json'))
ep = [r['ep'] for r in log]; tr = np.array([r['train_reward'] for r in log]) / 1e6
vx = [r['ep'] for r in log if 'val_reward' in r]; vy = np.array([r['val_reward'] for r in log if 'val_reward' in r]) / 1e6
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, tr, color='#b7d7c4', lw=0.8, label='Training day (with exploration)')
ma = np.convolve(tr, np.ones(10) / 10, mode='valid'); ax[0].plot(ep[9:], ma, color=COL['Q-learning agent'], lw=2, label='10-episode moving average')
ax[0].set_xlabel('Training episode (1 episode = 1 simulated day)'); ax[0].set_ylabel('Episode reward (Rs million)')
ax[0].set_title('Training reward curve'); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(vx, vy, color=COL['Q-learning agent'], lw=2, marker='o', ms=3, label='Q-learning (greedy, 6 validation days)')
for n, c in valb.items(): ax[1].axhline(c / 1e6, color=COL[n], ls='--', lw=1.3, label=n)
ax[1].set_xlabel('Training episode'); ax[1].set_ylabel('Mean validation reward (Rs million)'); ax[1].set_title('Greedy policy vs baselines'); ax[1].legend(frameon=False, fontsize=7, loc='lower right')
plt.tight_layout(); plt.savefig('fig_reward_curve.png'); plt.close()

names = list(res['policies'].keys())
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, (k, t, sc) in zip(ax, [('avg_wait', 'Avg passenger wait (min)', 1), ('rides', 'Rides completed / day', 1), ('fuel_litres', 'Fuel used / day (litres)', 1), ('cancel_rate', 'Cancelled requests (%)', 100)]):
    m = [res['policies'][n][k]['mean'] * sc for n in names]; sd = [res['policies'][n][k]['std'] * sc for n in names]
    a.bar(range(len(names)), m, yerr=sd, color=[COL[n] for n in names], capsize=2); a.set_xticks(range(len(names)))
    a.set_xticklabels(['Random', 'Busiest', 'Stay', 'Rule', 'Q-learn'], fontsize=8); a.set_title(t, fontsize=9)
    if k == 'avg_wait': a.axhline(7, color='k', ls=':', lw=1); a.text(4.4, 7.1, 'target 7 min', fontsize=7, ha='right')
    if k == 'rides': a.set_ylim(min(m) * 0.9, max(m) * 1.04)
plt.tight_layout(); plt.savefig('fig_metrics.png'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, n in zip(ax, ['Stay put (status quo)', 'Rule-based heuristic', 'Q-learning agent']):
    g = np.array(res['policies'][n]['zone_rate']).reshape(H, W)
    im = a.imshow(g, vmin=0.5, vmax=1.0, cmap='RdYlGn'); a.set_title(n + f"\nworst zone {g.min():.0%}", fontsize=9); a.set_xticks([]); a.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.015, label='Share of requests served'); plt.savefig('fig_fairness.png', bbox_inches='tight'); plt.close()
print(json.dumps({k: res[k] for k in ['day_types', 'paired', 'pilot', 'safety', 'state_count', 'val_baselines', 'fallback_share']}, indent=1))
for n in names: print(n, {k: round(res['policies'][n][k]['mean'], 3) for k in ['reward', 'avg_wait', 'rides', 'fuel_litres', 'cancel_rate', 'min_zone_rate', 'outskirt_rate', 'empty_share']})

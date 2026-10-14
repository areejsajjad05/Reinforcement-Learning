"""
SafarGo ride-hailing fleet simulator + Q-learning dispatch agent.
40 zones (8x5 grid, 1.5 km cells), 300 vehicles, one decision round every 5 minutes,
one episode = one operating day (06:00-24:00 = 216 steps).
"""
import math, random, bisect
import numpy as np

# ------------------------------------------------------------------ constants
W, H = 8, 5
NZ = W * H
NV = 300
NSTEPS = 216
STEP_MIN = 5
START_HOUR = 6
CELL_KM = 1.5
FARE_BASE, FARE_KM = 150.0, 45.0      # Rs
FUEL_RS_KM = 25.0                     # Rs per km (petrol ~Rs 270/L, ~11 km/L)
KM_PER_L = 11.0
LAMBDA_WAIT = 4.0                     # Rs penalty per passenger-minute waited
CANCEL_PEN = 100.0                    # Rs penalty per cancelled request
FAIR_BONUS = 120.0                    # Rs bonus x starvation index of the pickup zone
GAMMA = 0.95
RAIN_DEMAND = 1.3
TARGET_MEAN_DEMAND = 56.0             # mean requests / 5-min step (normal dry day)
DEST_DECAY = 4.0                      # trip-length decay (cells)
MAX_REPO = 8                          # cap on repositioning distance (cells)
N_ACTIONS = 4
ACTION_NAMES = ["Stay", "Nearby hotspot", "City hotspot", "Starved zone"]

# ------------------------------------------------------------------ geography
_rs = np.random.RandomState(11)
_types = ['res'] * 20 + ['com'] * 8 + ['uni'] * 4 + ['mkt'] * 4 + ['oth'] * 4
_rs.shuffle(_types)
ZTYPE = list(_types)
XY = [(z % W, z // W) for z in range(NZ)]
OUTSKIRT = [x in (0, W - 1) for x, y in XY]          # low-income / low-demand periphery
_TYPE_MULT = {'com': 1.6, 'uni': 1.4, 'mkt': 1.5, 'res': 0.8, 'oth': 1.0}
BASE = [(0.45 if OUTSKIRT[z] else 1.0) * float(_rs.lognormal(0, 0.3)) * _TYPE_MULT[ZTYPE[z]]
        for z in range(NZ)]
DIST = [[abs(XY[a][0] - XY[b][0]) + abs(XY[a][1] - XY[b][1]) for b in range(NZ)] for a in range(NZ)]
RING = [[[b for b in range(NZ) if DIST[a][b] == d] for d in range(3)] for a in range(NZ)]
NEAR = [[b for b in range(NZ) if 1 <= DIST[a][b] <= 2] for a in range(NZ)]
WITHIN4 = [[b for b in range(NZ) if 1 <= DIST[a][b] <= 4] for a in range(NZ)]


def _g(h, mu, sd):
    return math.exp(-0.5 * ((h - mu) / sd) ** 2)


def _prof(tp, h):
    if tp == 'res': return 0.5 + 1.3 * _g(h, 8, 1.2) + 0.9 * _g(h, 18.5, 1.8) + 0.5 * _g(h, 13, 2)
    if tp == 'com': return 0.4 + 0.5 * _g(h, 12.5, 2.5) + 1.7 * _g(h, 17.5, 1.3)
    if tp == 'uni': return 0.3 + 0.5 * _g(h, 8.5, 1.0) + 1.7 * _g(h, 14, 1.8) + 0.4 * _g(h, 17, 1.5)
    if tp == 'mkt': return 0.4 + 0.9 * _g(h, 13, 2) + 1.5 * _g(h, 19.5, 2.2)
    return 0.9 + 0.3 * _g(h, 10, 3)


def _destw(tp, h):
    if tp == 'com': return 0.6 + 1.6 * _g(h, 8.5, 1.5) + 0.4 * _g(h, 13, 1.5)
    if tp == 'res': return 0.5 + 1.8 * _g(h, 18.5, 2.0) + 0.5 * _g(h, 22, 2)
    if tp == 'uni': return 0.3 + 1.6 * _g(h, 8.5, 1.0)
    if tp == 'mkt': return 0.4 + 1.2 * _g(h, 17, 3)
    return 0.6


def _friday(tp, h):
    """Friday: quiet morning, dip around jummah prayer (12-14), post-prayer residential rebound."""
    m = 0.7 if h < 10 else 1.0
    if 12.0 <= h < 14.0 and tp in ('com', 'uni', 'oth'): m *= 0.5
    if 13.5 <= h < 15.0 and tp == 'res': m *= 1.5
    return m


HOUR = [START_HOUR + (t + 0.5) * STEP_MIN / 60.0 for t in range(NSTEPS + 4)]
_raw = np.array([[BASE[z] * _prof(ZTYPE[z], HOUR[t]) for z in range(NZ)] for t in range(NSTEPS + 4)])
_scale = TARGET_MEAN_DEMAND / _raw[:NSTEPS].sum(axis=1).mean()
RATE_BASE = _raw * _scale
RATE_FRI = np.array([[_friday(ZTYPE[z], HOUR[t]) for z in range(NZ)] for t in range(NSTEPS + 4)])
TIME_BUCKET = [min(5, int((HOUR[t] - START_HOUR) // 3)) for t in range(NSTEPS + 4)]

_DESTCDF = []
for hh in range(18):
    hrs = START_HOUR + hh + 0.5
    rows = []
    for o in range(NZ):
        cum, tot = [], 0.0
        for z in range(NZ):
            w = _destw(ZTYPE[z], hrs) * (BASE[z] ** 0.9) * math.exp(-DIST[o][z] / DEST_DECAY)
            if z == o: w *= 0.3
            tot += w
            cum.append(tot)
        rows.append(cum)
    _DESTCDF.append(rows)


def bucket_p(p):
    return 0 if p < 0.25 else (1 if p < 0.6 else (2 if p < 1.2 else 3))


N_STATES = 6 * 2 * 4 * 4 * 3 * 2 * 2   # time x weather x own-pressure x nearby-pressure x hotspot-dist x hotspot-hot x starved-near


# ------------------------------------------------------------------ policies
class StayPolicy:
    name = "Stay put (status quo)"
    def act(self, sim, i, z, t): return 0


class RandomPolicy:
    name = "Random action"
    def __init__(self, seed=0): self.r = random.Random(seed)
    def act(self, sim, i, z, t): return self.r.randrange(N_ACTIONS)


class BusiestPolicy:
    name = "Send to busiest zone"
    def act(self, sim, i, z, t): return 4


class HeuristicPolicy:
    """Hand-written rule: if my zone is oversupplied and a close zone is short of cars, go there."""
    name = "Rule-based heuristic (fallback)"
    def act(self, sim, i, z, t):
        P = sim.P
        if P[z] < 0.25:
            best = max((P[zz] for zz in NEAR[z]), default=0)
            if best > 0.8: return 1
            if any(sim.s[zz] > 0.3 for zz in WITHIN4[z]): return 3
        return 0


class QAgent:
    def __init__(self, alpha=0.2, eps=1.0, seed=0):
        self.Q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]
        self.N = [[0] * N_ACTIONS for _ in range(N_STATES)]
        self.alpha, self.eps = alpha, eps
        self.r = random.Random(seed)
        self.sample_update = None

    def choose(self, s, explore=True):
        if explore and self.r.random() < self.eps:
            return self.r.randrange(N_ACTIONS)
        q = self.Q[s]
        m = max(q)
        best = [a for a in range(N_ACTIONS) if q[a] == m]
        return best[0] if len(best) == 1 else self.r.choice(best)

    def arrays(self):
        return np.array(self.Q), np.array(self.N)


class QPolicy:
    """Greedy deployment policy with safety shield: unseen/rare states -> fallback heuristic."""
    name = "Q-learning agent"
    def __init__(self, Q, N, fallback=None, min_visits=30):
        self.Q, self.N = Q, N
        self.fallback = fallback or HeuristicPolicy()
        self.min_visits = min_visits
        self.fallback_uses = 0
        self.total_uses = 0
    def act(self, sim, i, z, t):
        s = sim.state_index(z, t)
        self.total_uses += 1
        if sum(self.N[s]) < self.min_visits:
            self.fallback_uses += 1
            return self.fallback.act(sim, i, z, t)
        q = self.Q[s]
        return max(range(N_ACTIONS), key=q.__getitem__)


# ------------------------------------------------------------------ simulator
class Sim:
    def __init__(self, seed=0, rain=False, friday=False, n_veh=NV):
        self.seed, self.rain, self.friday, self.nv = seed, rain, friday, n_veh
        self.rng = random.Random(seed)
        self.nprng = np.random.RandomState(seed)
        rates = RATE_BASE.copy()
        if rain: rates *= RAIN_DEMAND
        if friday: rates *= RATE_FRI
        self.rates = rates
        self.fnoise = self.nprng.lognormal(0, 0.15, NZ)      # imperfect demand forecast
        self.wx = 1 if rain else 0
        self.tripped = False
        self.trip_step = None

    # ---- features / targets (read by policies)
    def state_index(self, z, t):
        P = self.P
        ob = bucket_p(P[z])
        best = 0.0
        for zz in NEAR[z]:
            if P[zz] > best: best = P[zz]
        gz, gp = -1, -1.0
        for zz in range(NZ):
            if zz != z and P[zz] > gp: gp, gz = P[zz], zz
        d = DIST[z][gz]
        gdb = 0 if d <= 3 else (1 if d <= 6 else 2)
        gh = 1 if gp >= 0.6 else 0
        sb = 0
        S = self.s
        for zz in WITHIN4[z]:
            if S[zz] > 0.3: sb = 1; break
        idx = TIME_BUCKET[t]
        idx = idx * 2 + self.wx
        idx = idx * 4 + ob
        idx = idx * 4 + bucket_p(best)
        idx = idx * 3 + gdb
        idx = idx * 2 + gh
        idx = idx * 2 + sb
        return idx

    def target(self, a, z):
        if a == 0: return z
        P = self.P
        if a == 1:
            best, bz = -1.0, z
            for zz in NEAR[z]:
                if P[zz] > best: best, bz = P[zz], zz
            return bz
        if a == 2:
            best, bz = -1.0, z
            for zz in range(NZ):
                if zz != z and P[zz] > best: best, bz = P[zz], zz
            return bz if DIST[z][bz] <= MAX_REPO else z
        if a == 3:
            best, bz = 0.3, z
            for zz in WITHIN4[z]:
                if self.s[zz] > best: best, bz = self.s[zz], zz
            return bz
        D = self.D                                   # a == 4: busiest zone by raw demand
        bz = max(range(NZ), key=D.__getitem__)
        return bz

    # ---- main loop
    def run(self, ctrl, mask=None, base=None, learn=False, agent=None, fallback=None,
            breaker_wait=None, breaker_km=None, record_steps=False):
        rng, nprng = self.rng, self.nprng
        nv = self.nv
        rain = self.rain
        slow = 1.25 if rain else 1.0
        fmult = 1.1 if rain else 1.0
        base = base or StayPolicy()
        mask = [True] * nv if mask is None else list(mask)
        v_zone = [0] * nv
        wts = [BASE[z] if ZTYPE[z] == 'res' else 0.3 * BASE[z] for z in range(NZ)]
        for i in range(nv): v_zone[i] = rng.choices(range(NZ), weights=wts)[0]
        v_free = [0] * nv
        pend = [None] * nv                 # (state, action, t0, accumulated reward/100)
        self.s = [0.0] * NZ                # starvation index per zone
        queue = [[] for _ in range(NZ)]
        zreq = [0] * NZ; zserved = [0] * NZ; zcanc = [0] * NZ
        G = [dict(rides=0, wait=0.0, km_loaded=0.0, km_empty=0.0, revenue=0.0, fuel=0.0) for _ in range(2)]
        tot = dict(req=0, served=0, cancelled=0, wait_served=0.0, reward=0.0, fair=0.0, repos=0, decisions=0)
        waits = []
        series = dict(req=[], served=[], cancelled=[], wait=[], reward=[], idle=[]) if record_steps else None
        win = []                           # rolling window of pilot (wait_sum, n)
        rwin = []                          # rolling window of pilot repositioning km
        npilot = sum(1 for m in mask if m)
        self.tripped, self.trip_step = False, None
        self.max_repo = 0.0

        for t in range(NSTEPS):
            hh = int(HOUR[t]) - START_HOUR
            cnt = nprng.poisson(self.rates[t]).tolist()
            for z in range(NZ):
                c = cnt[z]
                if c:
                    queue[z].extend([t] * c); zreq[z] += c; tot['req'] += c
            idle = [[] for _ in range(NZ)]
            inbound = [0] * NZ
            for i in range(nv):
                f = v_free[i]
                if f <= t: idle[v_zone[i]].append(i)
                elif f <= t + 3: inbound[v_zone[i]] += 1
            # ---- matching (oldest requests first)
            reqs = [(b, z) for z in range(NZ) for b in queue[z]]
            rng.shuffle(reqs); reqs.sort(key=lambda x: x[0])
            newq = [[] for _ in range(NZ)]
            st_served = st_canc = 0; st_wait = 0.0; st_rew = 0.0
            pw, pn, cw, cn = 0.0, 0, 0.0, 0
            for b, z in reqs:
                age = t - b
                if age >= 3:
                    tot['cancelled'] += 1; zcanc[z] += 1; st_canc += 1
                    self.s[z] = 0.8 * self.s[z] + 0.2
                    st_rew -= CANCEL_PEN; waits.append(15.0); continue
                maxd = int((12 - 5 * age) // 5)
                veh = None; dd = 0
                for d in range(maxd + 1):
                    for zz in RING[z][d]:
                        if idle[zz]:
                            veh = idle[zz].pop(); dd = d; break
                    if veh is not None: break
                if veh is None:
                    newq[z].append(b); continue
                pick_min = 3 + 5 * dd
                wait = age * 5 + pick_min
                cum = _DESTCDF[hh][z]
                dest = bisect.bisect_left(cum, rng.random() * cum[-1])
                cells = max(1, DIST[z][dest])
                trip_steps = math.ceil(cells * slow)
                pick_steps = max(1, dd)
                km_pick = dd * CELL_KM if dd else CELL_KM / 2
                km_trip = cells * CELL_KM
                fare = FARE_BASE + FARE_KM * km_trip
                fuel = FUEL_RS_KM * (km_pick + km_trip) * fmult
                bonus = FAIR_BONUS * self.s[z]
                rew = fare - fuel - LAMBDA_WAIT * wait + bonus
                self.s[z] = 0.8 * self.s[z]
                v_zone[veh] = dest; v_free[veh] = t + pick_steps + trip_steps
                g = G[1 if mask[veh] else 0]
                g['rides'] += 1; g['wait'] += wait; g['km_loaded'] += km_trip; g['km_empty'] += km_pick
                g['revenue'] += fare; g['fuel'] += fuel
                tot['served'] += 1; tot['wait_served'] += wait; tot['fair'] += bonus
                zserved[z] += 1; st_served += 1; st_wait += wait; st_rew += rew; waits.append(wait)
                if mask[veh]: pw += wait; pn += 1
                else: cw += wait; cn += 1
                if learn and mask[veh] and pend[veh] is not None:
                    s0, a0, t0, r0 = pend[veh]; pend[veh] = (s0, a0, t0, r0 + rew / 100.0)
            queue = newq
            # ---- circuit breaker on the pilot group
            win.append((pw, pn, cw, cn))
            if len(win) > 12: win.pop(0)
            if breaker_wait and not self.tripped and t >= 12:
                sw = sum(x[0] for x in win); sn = sum(x[1] for x in win)
                cw_ = sum(x[2] for x in win); cn_ = sum(x[3] for x in win)
                # with a control group: trip if pilot wait exceeds control wait by `breaker_wait` minutes;
                # with no control group (full roll-out): treat `breaker_wait` as an absolute limit
                lim = (cw_ / cn_ + breaker_wait) if (npilot < nv and cn_ >= 15) else (breaker_wait if npilot >= nv else None)
                if sn >= 15 and lim is not None and sw / sn > lim:
                    self.tripped, self.trip_step = True, t
            if breaker_km and not self.tripped and len(rwin) >= 6 and sum(rwin) > breaker_km * npilot:
                self.tripped, self.trip_step = True, t
            # ---- pressure signals
            fc = [(self.rates[t + 1][z] + self.rates[t + 2][z]) * self.fnoise[z] for z in range(NZ)]
            qlen = [len(queue[z]) for z in range(NZ)]
            supply = [len(idle[z]) + inbound[z] for z in range(NZ)]
            self.D = [qlen[z] + fc[z] for z in range(NZ)]
            self.P = [self.D[z] / (supply[z] + 0.5) for z in range(NZ)]
            ids = [i for z in range(NZ) for i in idle[z]]
            rng.shuffle(ids)
            prk = 0.0
            for i in ids:
                z = v_zone[i]
                ctl = mask[i]
                tot['decisions'] += 1
                if learn and ctl:
                    s = self.state_index(z, t)
                    if pend[i] is not None:
                        s0, a0, t0, r0 = pend[i]
                        k = t - t0
                        target = r0 + (GAMMA ** k) * max(agent.Q[s])
                        old = agent.Q[s0][a0]
                        agent.Q[s0][a0] = old + agent.alpha * (target - old)
                        if agent.sample_update is None and k >= 3 and r0 > 1.0 and t > 100:
                            agent.sample_update = dict(s=s0, a=a0, k=k, r=r0, q_old=old, max_next=max(agent.Q[s]),
                                                       q_new=agent.Q[s0][a0], alpha=agent.alpha)
                    a = agent.choose(s, explore=True)
                    agent.N[s][a] += 1
                    pend[i] = (s, a, t, 0.0)
                else:
                    pol = (base if not ctl else (fallback if (self.tripped and fallback) else ctrl))
                    a = pol.act(self, i, z, t)
                tz = self.target(a, z)
                if tz != z:
                    d = DIST[z][tz]
                    steps = max(1, math.ceil(d * slow))
                    km = d * CELL_KM
                    cost = km * FUEL_RS_KM * fmult
                    v_zone[i] = tz; v_free[i] = t + steps
                    supply[z] -= 1; supply[tz] += 1
                    self.P[z] = self.D[z] / (supply[z] + 0.5); self.P[tz] = self.D[tz] / (supply[tz] + 0.5)
                    g = G[1 if ctl else 0]
                    g['km_empty'] += km; g['fuel'] += cost
                    if ctl: prk += km
                    tot['repos'] += 1; st_rew -= cost
                    if learn and ctl: s0, a0, t0, r0 = pend[i]; pend[i] = (s0, a0, t0, r0 - cost / 100.0)
            rwin.append(prk)
            if len(rwin) > 12: rwin.pop(0)
            if npilot and len(rwin) >= 6: self.max_repo = max(self.max_repo, sum(rwin) / npilot)
            tot['reward'] += st_rew
            if series is not None:
                series['req'].append(sum(cnt)); series['served'].append(st_served); series['cancelled'].append(st_canc)
                series['wait'].append(st_wait / st_served if st_served else 0.0)
                series['reward'].append(st_rew); series['idle'].append(len(ids))
        return self._summarise(tot, G, zreq, zserved, waits, series)

    def _summarise(self, tot, G, zreq, zserved, waits, series):
        km_l = G[0]['km_loaded'] + G[1]['km_loaded']
        km_e = G[0]['km_empty'] + G[1]['km_empty']
        fuel_rs = G[0]['fuel'] + G[1]['fuel']
        rev = G[0]['revenue'] + G[1]['revenue']
        rates = [zserved[z] / zreq[z] for z in range(NZ) if zreq[z] > 0]
        out_r = [zserved[z] / max(1, zreq[z]) for z in range(NZ) if OUTSKIRT[z]]
        cen_r = [zserved[z] / max(1, zreq[z]) for z in range(NZ) if not OUTSKIRT[z]]
        res = dict(
            reward=tot['reward'], requests=tot['req'], rides=tot['served'], cancelled=tot['cancelled'],
            avg_wait=tot['wait_served'] / max(1, tot['served']),
            p90_wait=float(np.percentile(waits, 90)) if waits else 0.0,
            cancel_rate=tot['cancelled'] / max(1, tot['req']),
            km_total=km_l + km_e, km_empty=km_e, empty_share=km_e / max(1e-9, km_l + km_e),
            fuel_litres=(km_l + km_e) / KM_PER_L * (1.1 if self.rain else 1.0),
            fuel_cost=fuel_rs, revenue=rev, net_revenue=rev - fuel_rs, repos=tot['repos'],
            min_zone_rate=min(rates), zones_below_70=sum(r < 0.7 for r in rates) / len(rates),
            outskirt_rate=float(np.mean(out_r)), central_rate=float(np.mean(cen_r)),
            tripped=self.tripped, trip_step=self.trip_step, max_repo=self.max_repo, decisions=tot['decisions'],
            zone_rate=[zserved[z] / max(1, zreq[z]) for z in range(NZ)],
        )
        for gi, name in ((0, 'ctrl_off'), (1, 'ctrl_on')):
            g = G[gi]
            res[name] = dict(rides=g['rides'], avg_wait=g['wait'] / max(1, g['rides']),
                             net_revenue=g['revenue'] - g['fuel'], km_empty=g['km_empty'],
                             km_total=g['km_loaded'] + g['km_empty'])
        if series is not None: res['series'] = series
        return res


def day_config(seed):
    r = random.Random(seed * 7919 + 13)
    return dict(rain=r.random() < 0.27, friday=r.random() < 0.17)

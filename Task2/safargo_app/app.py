import json, numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sim import *

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="SafarGo RL Dispatch – MVP", page_icon="🚕", layout="wide")


@st.cache_resource
def load():
    Q = np.load(HERE / "q_table.npy"); N = np.load(HERE / "visits.npy")
    return Q.tolist(), N.tolist()


Q, N = load()
st.title("🚕 SafarGo – RL Fleet Repositioning (MVP)")
st.caption("Q-learning dispatch agent on a simulated Rawalpindi–Islamabad fleet: 300 vehicles, 40 zones, decisions every 5 minutes. "
           "Reinforcement Learning assignment – PMAS-AAUR (UIIT).")

tab_sim, tab_res, tab_risk = st.tabs(["▶ Run a simulated day", "📊 Training & test results", "⚖ Ethics & risks"])

with tab_sim:
    c1, c2, c3 = st.columns(3)
    with c1:
        policy_name = st.selectbox("Policy to test", ["Q-learning agent", "Rule-based heuristic (fallback)", "Send to busiest zone", "Random action", "Stay put (status quo)"])
        seed = st.number_input("Random seed (demand scenario)", 0, 99999, 5001)
    with c2:
        rain = st.checkbox("Rain (demand +30%, slower trips)", value=False)
        friday = st.checkbox("Friday (jummah-prayer demand pattern)", value=False)
    with c3:
        pilot_pct = st.slider("Share of fleet controlled by the agent", 0, 100, 10, 10, help="The company can only test on 10% of vehicles. The rest stay put (status quo).")
        breaker = st.checkbox("Safety circuit breaker → rule-based fallback", value=True)

    if st.button("Run simulation", type="primary"):
        n_pilot = int(NV * pilot_pct / 100)
        mask = [i < n_pilot for i in range(NV)]
        pols = {"Q-learning agent": QPolicy(Q, N, min_visits=30), "Rule-based heuristic (fallback)": HeuristicPolicy(),
                "Send to busiest zone": BusiestPolicy(), "Random action": RandomPolicy(3), "Stay put (status quo)": StayPolicy()}
        pol = pols[policy_name]
        with st.spinner("Simulating 18 operating hours…"):
            base = Sim(seed=int(seed), rain=rain, friday=friday).run(StayPolicy(), record_steps=True)
            kw = dict(mask=mask, base=StayPolicy(), record_steps=True)
            if breaker and 0 < n_pilot:
                kw.update(fallback=HeuristicPolicy(), breaker_km=7.0, breaker_wait=3.0 if n_pilot < NV else 12.0)
            r = Sim(seed=int(seed), rain=rain, friday=friday).run(pol, **kw)

        st.subheader(f"Result: {policy_name} on {pilot_pct}% of the fleet ({n_pilot} vehicles)")
        m = st.columns(5)
        m[0].metric("Avg wait (min)", f"{r['avg_wait']:.2f}", f"{r['avg_wait'] - base['avg_wait']:+.2f} vs status quo", delta_color="inverse")
        m[1].metric("Rides completed", f"{r['rides']:,}", f"{r['rides'] - base['rides']:+,}")
        m[2].metric("Fuel used (L)", f"{r['fuel_litres']:,.0f}", f"{r['fuel_litres'] - base['fuel_litres']:+,.0f}", delta_color="inverse")
        m[3].metric("Cancelled (%)", f"{100 * r['cancel_rate']:.1f}", f"{100 * (r['cancel_rate'] - base['cancel_rate']):+.1f} pts", delta_color="inverse")
        m[4].metric("Worst-zone service rate", f"{100 * r['min_zone_rate']:.0f}%", f"{100 * (r['min_zone_rate'] - base['min_zone_rate']):+.0f} pts")
        if n_pilot and r["tripped"]:
            st.error(f"⚠ Circuit breaker tripped at {START_HOUR + r['trip_step'] * 5 / 60:.1f}h – pilot vehicles switched to the rule-based fallback for the rest of the day.")
        elif n_pilot and breaker:
            st.success("Circuit breaker did not trip – pilot behaved within safety limits.")
        if 0 < n_pilot < NV:
            a, b = r["ctrl_on"], r["ctrl_off"]
            st.markdown("**Pilot A/B inside the same fleet** (per vehicle, same day)")
            st.dataframe(pd.DataFrame({
                "Group": [f"Agent-controlled ({n_pilot})", f"Control – status quo ({NV - n_pilot})"],
                "Rides / vehicle": [round(a["rides"] / n_pilot, 1), round(b["rides"] / (NV - n_pilot), 1)],
                "Avg wait (min)": [round(a["avg_wait"], 2), round(b["avg_wait"], 2)],
                "Net revenue / vehicle (Rs)": [round(a["net_revenue"] / n_pilot), round(b["net_revenue"] / (NV - n_pilot))],
                "Empty km / vehicle": [round(a["km_empty"] / n_pilot, 1), round(b["km_empty"] / (NV - n_pilot), 1)]}), hide_index=True, width="stretch")
        s, sb = r["series"], base["series"]
        hrs = [START_HOUR + h for h in range(18)]
        agg = lambda x, f=np.sum: [f(x[h * 12:(h + 1) * 12]) for h in range(18)]
        df = pd.DataFrame({"hour": hrs, "Requests": agg(s["req"]), "Rides served": agg(s["served"]), "Cancelled": agg(s["cancelled"]),
                           "Wait – this policy (min)": agg(s["wait"], np.mean), "Wait – status quo (min)": agg(sb["wait"], np.mean)}).set_index("hour")
        l, rr = st.columns(2)
        l.markdown("**Demand and service by hour**"); l.bar_chart(df[["Requests", "Rides served", "Cancelled"]])
        rr.markdown("**Average wait by hour (min)**"); rr.line_chart(df[["Wait – this policy (min)", "Wait – status quo (min)"]])
        fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
        for a_, rs_, ttl in zip(ax, [base, r], ["Status quo", policy_name]):
            im = a_.imshow(np.array(rs_["zone_rate"]).reshape(H, W), vmin=0.4, vmax=1, cmap="RdYlGn"); a_.set_title(f"{ttl}: share of requests served", fontsize=8); a_.set_xticks([]); a_.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.025); st.pyplot(fig)
        st.caption("Zones in columns 1 and 8 are the low-demand, lower-income periphery – the areas a profit-only agent would ignore.")
    else:
        st.info("Choose a scenario and press **Run simulation**. Try the Q-learning agent on 10% of the fleet (the pilot), then on 100%, then switch to 'Send to busiest zone' to see how a naive policy fails.")

with tab_res:
    res = json.load(open(HERE / "results.json"))
    st.markdown("### Held-out test: 30 unseen simulated days (mean per day)")
    rows = []
    for n, v in res["policies"].items():
        rows.append({"Policy": n, "Reward (Rs)": round(v["reward"]["mean"]), "Avg wait (min)": round(v["avg_wait"]["mean"], 2), "Rides": round(v["rides"]["mean"]),
                     "Fuel (L)": round(v["fuel_litres"]["mean"]), "Cancelled %": round(100 * v["cancel_rate"]["mean"], 1), "Worst zone served %": round(100 * v["min_zone_rate"]["mean"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.image(str(HERE / "fig_reward_curve.png"), caption="Reward curve")
    st.image(str(HERE / "fig_metrics.png"), caption="Waiting time, rides completed, fuel use and cancellations (error bars = day-to-day std. dev.)")
    st.image(str(HERE / "fig_fairness.png"), caption="Fairness: share of requests served per zone")
    p = res["pilot"]
    st.markdown("### 10% pilot (30 agent vehicles vs 270 control vehicles, same days)")
    st.write(f"Agent cars: {p['ctrl_on']['rides_per_veh']:.1f} rides/vehicle, net Rs {p['ctrl_on']['net_per_veh']:,.0f}/vehicle/day, {p['ctrl_on']['km_empty_per_veh']:.0f} empty km. "
             f"Control cars: {p['ctrl_off']['rides_per_veh']:.1f} rides/vehicle, net Rs {p['ctrl_off']['net_per_veh']:,.0f}/vehicle/day, {p['ctrl_off']['km_empty_per_veh']:.0f} empty km. "
             f"Net-revenue difference: Rs {p['net_diff']['mean']:,.0f}/vehicle/day (95% CI {p['net_diff']['lo']:,.0f} to {p['net_diff']['hi']:,.0f}).")

with tab_risk:
    st.markdown("""
**Unfair treatment of poorer / low-demand areas.** A pure revenue agent learns to abandon the periphery. We (1) add a starvation-based fairness bonus to the reward, (2) include a dedicated *'send to starved zone'* action, and (3) monitor the worst-zone service rate as a hard KPI – not just the average.

**Driver trust.** In our pilot the agent's cars drive more empty kilometres and earn *less* than control cars. Fix before any real pilot: pay a per-km repositioning allowance, make recommendations explainable ('Zone 14 is short of cars'), let drivers decline, and share the pilot's results openly.

**Data privacy.** Use zone-level aggregates (no rider IDs or exact GPS traces) in the state, store ride logs with pseudonymised IDs, delete raw GPS after aggregation, and comply with Pakistan's personal-data-protection requirements.

**Operational risk.** Run on 10% of vehicles only, keep the rule-based fallback, and trip a circuit breaker when pilot cars burn too much fuel or wait times rise relative to the control group.
""")

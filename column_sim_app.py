# column_sim_app.py
"""
Mini Dynamic Distillation Column + PFD sketch (Streamlit)
- Simple dynamic mass-balance model with N stages (CSTR-like)
- Overhead condenser -> reflux drum -> reflux pump (reflux + distillate)
- Reboiler -> bottom pump (bottoms product)
- Uses a simple relative-volatility equilibrium: y = alpha*x / (1 + (alpha-1)*x)
Notes:
- This is a teaching/training model, simplified for qualitative behavior.
- Units: flow in m3/hr, time in seconds for simulation steps.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import time

st.set_page_config(layout="wide", page_title="Distillation Column Practice App")
st.title("Distillation Column — Mini DCS Practice (Column + Condenser + Reflux Drum + Pumps)")

# -----------------------
# Sidebar: model inputs
# -----------------------
st.sidebar.header("Column & Feed configuration")
n_stages = st.sidebar.slider("Number of stages (including reboiler & condenser stages)", 4, 30, 10)
feed_stage = st.sidebar.slider("Feed stage (1 = top stage)", 1, n_stages, int(n_stages/2))
feed_flow = st.sidebar.number_input("Feed flow (m³/hr)", value=5.0, step=0.1)
feed_comp = st.sidebar.slider("Feed mole fraction of light key (xF)", 0.0, 1.0, 0.4, 0.01)
feed_quality = st.sidebar.selectbox("Feed quality (liquid/vapor)", options=["Saturated liquid (q=1)", "Saturated vapor (q=0)"])
q = 1.0 if feed_quality.startswith("Saturated liquid") else 0.0

st.sidebar.header("Thermo & Operation")
alpha = st.sidebar.number_input("Relative volatility α (light/heavy)", value=2.0, step=0.1)
reflux_ratio = st.sidebar.slider("Reflux ratio R (L/D)", 0.1, 10.0, 2.0, 0.1)
boilup_ratio = st.sidebar.slider("Boilup ratio (V/B) relative)", 1.2, 10.0, 2.0, 0.1)

st.sidebar.header("Equipment & Controller")
reflux_drum_volume = st.sidebar.number_input("Reflux drum holdup (m³)", 0.05, 5.0, 0.5, 0.01)
stage_holdup = st.sidebar.number_input("Stage holdup (m³ per stage)", 0.01, 1.0, 0.1, 0.01)
reboiler_heat = st.sidebar.number_input("Reboiler duty (arbitrary units)", 10.0, 1000.0, 100.0, 1.0)

st.sidebar.header("Simulation controls")
sim_time = st.sidebar.number_input("Sim time (s)", value=600, step=10)
dt = st.sidebar.number_input("Time step dt (s)", value=1.0, step=0.1)
sim_speed_ms = st.sidebar.slider("UI update delay per step (ms)", 0, 200, 0)

# Buttons
start_button = st.sidebar.button("Start Simulation ▶")
stop_button = st.sidebar.button("Stop ⏹")
reset_button = st.sidebar.button("Reset")

# -----------------------
# Model functions
# -----------------------

def equilibrium_y(x, alpha):
    """Simple vapor-liquid equilibrium relation (approximate)."""
    # ensure bounds
    x = np.clip(x, 1e-9, 1-1e-9)
    y = (alpha * x) / (1 + (alpha - 1) * x)
    y = np.clip(y, 1e-9, 1-1e-9)
    return y

def initialize_states(n, feed_x, feed_flow, feed_stage, stage_holdup, reflux_drum_volume):
    """Initialize stage liquid compositions, flows, drum, etc."""
    # liquid composition in each stage (mole fraction of light key)
    x = np.ones(n) * feed_x
    # initial liquid flow L (m3/hr) default equal feed_flow
    L = np.ones(n+1) * feed_flow  # L between stages (L0 = reflux ; Ln = bottoms return)
    V = np.ones(n+1) * feed_flow  # vapor flows
    # holdups
    M = np.ones(n) * stage_holdup
    drum = {'holdup': reflux_drum_volume, 'x': feed_x}
    return x, L, V, M, drum

# Simulation state stored in session
if 'running' not in st.session_state or reset_button:
    st.session_state['running'] = False
    st.session_state['time'] = 0.0
    st.session_state['x'], st.session_state['L'], st.session_state['V'], st.session_state['M'], st.session_state['drum'] = \
        initialize_states(n_stages, feed_comp, feed_flow, feed_stage, stage_holdup, reflux_drum_volume)
    st.session_state['history'] = {'time':[], 'x_mean':[], 'overhead_x':[], 'bottoms_x':[], 'D':[], 'B':[]}

# update stage count or feed changes: reinitialize (but preserve history if not reset)
if len(st.session_state['x']) != n_stages:
    st.session_state['x'], st.session_state['L'], st.session_state['V'], st.session_state['M'], st.session_state['drum'] = \
        initialize_states(n_stages, feed_comp, feed_flow, feed_stage, stage_holdup, reflux_drum_volume)

# Controls derived: for a simple model, choose D (distillate) and B (bottoms) partitioning from feed and reflux ratio
# We'll compute an instantaneous D and B for mass balance: Feed = D + B (volumetric simplification).
# For dynamics, flows L and V are set from reflux ratio and boilup approximate relationships.

# Stop logic
if stop_button:
    st.session_state['running'] = False

if start_button:
    st.session_state['running'] = True

# Reset
if reset_button:
    st.session_state['x'], st.session_state['L'], st.session_state['V'], st.session_state['M'], st.session_state['drum'] = \
        initialize_states(n_stages, feed_comp, feed_flow, feed_stage, stage_holdup, reflux_drum_volume)
    st.session_state['history'] = {'time':[], 'x_mean':[], 'overhead_x':[], 'bottoms_x':[], 'D':[], 'B':[]}
    st.session_state['time'] = 0.0
    st.session_state['running'] = False

# -----------------------
# Simulation step
# -----------------------
def sim_step(dt):
    n = n_stages
    x = st.session_state['x']
    L = st.session_state['L']
    V = st.session_state['V']
    M = st.session_state['M']
    drum = st.session_state['drum']

    # Very simple operational flows:
    # Choose distillate volumetric D and bottoms B from feed and reflux ratio R:
    # Let D = feed_flow / (1 + (some factor depending on reflux)) — this is simplistic
    # Instead we choose D as a fraction based on reflux_ratio:
    D = feed_flow * (1.0 / (1.0 + 1.0/reflux_ratio)) * 0.5  # crude guess to make D vary with R
    B = max(0.001, feed_flow - D)

    # Liquid and vapor flows between stages (simplified)
    # Top reflux L0 = R * D
    L0 = reflux_ratio * D
    # Boilup Vn = boilup_ratio * B
    Vn = boilup_ratio * B

    # set internal flows roughly constant down the column
    # L_i ~ L0 (downwards), V_i ~ Vn (upwards)
    L_internal = np.ones(n+1) * max(1e-6, L0)
    V_internal = np.ones(n+1) * max(1e-6, Vn)

    # Save flows
    L[:] = L_internal
    V[:] = V_internal

    # Mass balances on each stage (CSTR-like dynamic stage)
    # d(x_i*M_i)/dt = L_{i+1}*x_{i+1} + V_{i-1}*y_{i-1} - L_i*x_i - V_i*y_i + feed_injected_on_feed_stage
    x_new = x.copy()
    y = equilibrium_y(x, alpha)  # vapor compositions on each stage

    for i in range(n):
        # flows indices: L[i] is downflow leaving stage i to i+1 (L0 is reflux to top stage)
        L_up = L[i]     # from stage above i-1 -> i (L0 for top)
        L_down = L[i+1] # leaving to below stage
        V_up = V[i]     # vapor from below (i+1) -> i
        V_down = V[i+1] # vapor leaving stage i -> above

        # vapor compositions used:
        y_up = y[i-1] if i-1 >= 0 else equilibrium_y(x[0], alpha)  # for top stage approximation
        y_down = y[i]  # vapor leaving this stage upward

        # feed term
        feed_term = 0.0
        if (i+1) == feed_stage:  # feed_stage uses 1-based indexing
            # feed goes into this stage as liquid (if q=1) or vapor (q=0)
            if q >= 1.0:
                feed_term += feed_flow * feed_comp
            else:
                # if vapor feed, it enters as vapor adding to V flows (approximate)
                # represent as adding to V_up composition -> simple equivalent mass to stage
                pass

        # compute net molar change of light key in liquid on stage i (m3/hr * mole fraction)
        # Use flows in m3/hr and assume concentrations proportional to mole fractions (qualitative)
        inflow_light = L_up * (x[i-1] if i-1 >= 0 else drum['x']) + V_up * y_up
        outflow_light = L_down * x[i] + V_down * y_down
        # convert to time derivative of x: dx/dt = (in - out) / M_i
        dxdt = (inflow_light - outflow_light + feed_term) / M[i]
        x_new[i] = x[i] + dxdt * (dt / 3600.0)  # convert flows m3/hr into m3/s via /3600 for dt in s

        # keep within bounds
        x_new[i] = float(np.clip(x_new[i], 1e-6, 1-1e-6))

    # Reflux drum dynamic: receives condensed vapor (assume composition y_top) and splits to reflux and distillate
    y_top = equilibrium_y(x[0], alpha)
    # condensed flow = V_internal[0] (vapor leaving top condenses)
    condensed = V_internal[0]
    # drum holdup update: simple integrator
    drum['x'] = (drum['x'] * drum['holdup'] + condensed * y_top * dt/3600.0) / (drum['holdup'] + (condensed * dt/3600.0))
    # split: reflux fraction fr =  R / (R + 1) roughly, so reflux = R*D ; D came from above
    reflux = L0
    # Remove distillate outflow:
    distillate = D
    # Update drum holdup (no complex dynamics for level here)
    # (In a better model you'd have level controller; here holdup stays constant)

    # Reboiler stage (bottom): simple boilup adds vapor
    # Vapor leaving bottom Vn = boilup
    # bottom composition approx x[-1]
    # remove bottoms B
    bottoms = B

    # update session state
    st.session_state['x'] = x_new
    st.session_state['L'] = L
    st.session_state['V'] = V
    st.session_state['drum'] = drum

    # log history
    st.session_state['history']['time'].append(st.session_state['time'])
    st.session_state['history']['x_mean'].append(np.mean(x_new))
    st.session_state['history']['overhead_x'].append(drum['x'])
    st.session_state['history']['bottoms_x'].append(x_new[-1])
    st.session_state['history']['D'].append(distillate)
    st.session_state['history']['B'].append(bottoms)

    st.session_state['time'] += dt

# -----------------------
# Run simulation loop (nonblocking)
# -----------------------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Column PFD sketch")
    # draw a simple PFD with matplotlib
    fig, ax = plt.subplots(figsize=(6,8))
    ax.axis('off')

    # column rectangle
    col_x = 0.35
    col_w = 0.3
    col_h = 0.8
    ax.add_patch(Rectangle((col_x, 0.1), col_w, col_h, fill=False, linewidth=2))

    # draw trays as horizontal lines
    for i in range(n_stages):
        y = 0.1 + (i+1) * (col_h / (n_stages+1))
        ax.hlines(y, col_x, col_x + col_w, colors='gray', linewidth=1)

    # condenser on top
    ax.add_patch(Rectangle((0.1, 0.9), 0.15, 0.08, fill=False))
    ax.text(0.175, 0.935, "Condenser", ha='center', va='center')

    # reflux drum on right
    ax.add_patch(Rectangle((0.7, 0.85), 0.18, 0.12, fill=False))
    ax.text(0.79, 0.905, "Reflux\nDrum", ha='center', va='center')

    # reboiler at bottom
    ax.add_patch(Rectangle((0.1, 0.02), 0.15, 0.07, fill=False))
    ax.text(0.175, 0.06, "Reboiler", ha='center', va='center')

    # pumps (bottom and reflux pumps)
    ax.add_patch(Rectangle((0.7, 0.05), 0.12, 0.06, fill=False))
    ax.text(0.76, 0.08, "Bottom\nPump", ha='center', va='center', fontsize=8)

    ax.add_patch(Rectangle((0.7, 0.6), 0.12, 0.06, fill=False))
    ax.text(0.76, 0.63, "Reflux\nPump", ha='center', va='center', fontsize=8)

    # stream arrows
    # from condenser to drum
    ax.add_patch(FancyArrow(0.25, 0.95, 0.45, 0.0, width=0.01, length_includes_head=True))
    ax.text(0.65, 0.96, "Condensed vapor →", ha='left', va='bottom')
    # from drum to top (reflux)
    ax.add_patch(FancyArrow(0.65, 0.82, -0.25, -0.05, width=0.01, length_includes_head=True))
    ax.text(0.45, 0.78, "Reflux", ha='center', va='center')
    # distillate out from drum
    ax.add_patch(FancyArrow(0.85, 0.85, 0.1, 0.0, width=0.01, length_includes_head=True))
    ax.text(0.97, 0.86, "Distillate out", ha='left', va='center', fontsize=9)

    # vapor up from reboiler to column
    ax.add_patch(FancyArrow(0.175, 0.095, 0.0, 0.7, width=0.01, length_includes_head=True))
    ax.text(0.22, 0.5, "Vapor up", ha='left', va='center')

    # bottoms out via bottom pump
    ax.add_patch(FancyArrow(0.85, 0.07, 0.1, 0.0, width=0.01, length_includes_head=True))
    ax.text(0.98, 0.09, "Bottoms out", ha='left', va='center', fontsize=9)

    ax.set_title("PFD: Column + Condenser + Reflux Drum + Pumps (schematic)")
    st.pyplot(fig)

with col2:
    st.subheader("Quick status / controls")
    st.metric("Time (s)", f"{st.session_state['time']:.1f}")
    # show top & bottom composition and flows
    hx = st.session_state['history']
    if len(hx['time'])>0:
        st.metric("Overhead x (drum)", f"{hx['overhead_x'][-1]:.3f}")
        st.metric("Bottoms x (liquid)", f"{hx['bottoms_x'][-1]:.3f}")
        st.metric("Distillate flow D (m3/hr)", f"{hx['D'][-1]:.3f}")
    else:
        st.metric("Overhead x (drum)", f"{st.session_state['drum']['x']:.3f}")
        st.metric("Bottoms x (liquid)", f"{st.session_state['x'][-1]:.3f}")
        st.metric("Distillate flow D (m3/hr)", "—")

# execute simulation steps if running
if st.session_state['running']:
    steps = int(sim_time / dt)
    for i in range(steps):
        if not st.session_state['running']:
            break
        sim_step(dt)
        # update UI plots occasionally
        if (i % max(1, int(5/dt))) == 0:
            time_history = st.session_state['history']['time']
            df_hist = pd.DataFrame(st.session_state['history'])
            # plots
            st.subheader("Column composition trends")
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(df_hist.rename(columns={'time':'index'}).set_index('time')[['x_mean','overhead_x','bottoms_x']])
            with c2:
                st.line_chart(df_hist.rename(columns={'time':'index'}).set_index('time')[['D','B']])
            # slight UI delay
            if sim_speed_ms > 0:
                time.sleep(sim_speed_ms/1000.0)
    st.session_state['running'] = False
    st.success("Simulation run completed (or stopped).")

# Final results view & allow export
st.subheader("Results & export")
hist = st.session_state['history']
if len(hist['time']) > 0:
    df = pd.DataFrame(hist)
    st.dataframe(df.tail(50))
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "column_sim_results.csv", "text/csv")
else:
    st.info("No simulation data yet — press Start Simulation to run.")

st.markdown("---")
st.markdown("**Notes:** This model uses simplified mass-balance and a very approximate equilibrium relation. "
            "It is for training/qualitative exploration only. For rigorous design or control studies, use validated VLE, enthalpy balances, and a process simulator (Aspen/DWSIM).")

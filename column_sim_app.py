import streamlit as st
import numpy as np
import pandas as pd

# Safe import for matplotlib
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    import subprocess
    subprocess.run(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

# ------------------------
# PROCESS SIMULATION MODEL
# ------------------------

def simulate_column(feed_temp, feed_flow, reflux_ratio, stages=10):
    """Simple distillation column model."""
    stage_temps = np.linspace(feed_temp + 20, feed_temp - 30, stages)
    stage_pressures = np.linspace(2.0, 1.0, stages)  # bar
    
    # Overhead product purity (simple relationship)
    top_purity = np.clip(0.8 + (reflux_ratio - 1) * 0.05, 0, 0.99)
    bottom_purity = 1 - top_purity
    
    return stage_temps, stage_pressures, top_purity, bottom_purity

# ------------------------
# STREAMLIT APP UI
# ------------------------

st.title("🛠️ Mini Process Plant Simulator")
st.markdown("""
This is a **simple interactive simulation** of a distillation column with:
- Bottom pump
- Overhead cooler
- Reflux drum
""")

st.sidebar.header("🔧 Process Inputs")
feed_temp = st.sidebar.slider("Feed Temperature (°C)", 50, 150, 100)
feed_flow = st.sidebar.slider("Feed Flowrate (kmol/hr)", 10, 200, 100)
reflux_ratio = st.sidebar.slider("Reflux Ratio", 0.5, 5.0, 1.5)

# Run simulation
temps, pressures, top_x, bottom_x = simulate_column(feed_temp, feed_flow, reflux_ratio)

# ------------------------
# DISPLAY RESULTS
# ------------------------

col1, col2 = st.columns(2)
with col1:
    st.metric("Overhead Purity", f"{top_x*100:.2f}%")
    st.metric("Bottom Purity", f"{bottom_x_

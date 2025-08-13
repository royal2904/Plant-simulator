# plant_sim.py
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Mini Plant Process Simulator", layout="wide")

st.title("🛠️ Mini Process Plant Simulator")
st.write("Adjust pump, valve, heater, and reactor parameters to see process behavior in real-time.")

# Sidebar controls
st.sidebar.header("Equipment Controls")

# Pump
pump_flow = st.sidebar.slider("Pump Flowrate (m³/h)", 0.0, 100.0, 50.0)

# Valve
valve_opening = st.sidebar.slider("Valve Opening (%)", 0, 100, 50)

# Heater
heater_temp = st.sidebar.slider("Heater Outlet Temp (°C)", 20, 300, 150)

# Reactor
reaction_rate = st.sidebar.slider("Reaction Rate Constant k (1/min)", 0.0, 1.0, 0.3)
feed_concentration = st.sidebar.slider("Feed Concentration (mol/L)", 0.0, 5.0, 2.0)

# Column
reflux_ratio = st.sidebar.slider("Reflux Ratio", 0.5, 5.0, 2.0)

# Process simulation
# Calculate flow after valve
flow_after_valve = pump_flow * (valve_opening / 100)

# Calculate reactor outlet concentration (first-order decay)
residence_time = 5.0  # min
reactor_out_conc = feed_concentration * np.exp(-reaction_rate * residence_time)

# Heater effect
outlet_temp = heater_temp

# Column separation (very simplified)
distillate_conc = reactor_out_conc / reflux_ratio
bottoms_conc = reactor_out_conc - distillate_conc

# Make a dataframe for display
data = {
    "Equipment": ["Pump", "Valve", "Heater", "Reactor", "Column"],
    "Parameter": [
        f"Flowrate = {pump_flow:.1f} m³/h",
        f"Opening = {valve_opening}%",
        f"Outlet Temp = {outlet_temp:.1f} °C",
        f"Outlet Conc. = {reactor_out_conc:.2f} mol/L",
        f"Reflux Ratio = {reflux_ratio:.2f}"
    ],
    "Effect": [
        f"Flow after valve: {flow_after_valve:.1f} m³/h",
        f"Flow reduction: {(100 - valve_opening)}%",
        f"Temperature to reactor: {outlet_temp:.1f} °C",
        f"Conversion: {(1 - reactor_out_conc/feed_concentration)*100:.1f}%",
        f"Distillate Conc: {distillate_conc:.2f}, Bottoms Conc: {bottoms_conc:.2f}"
    ]
}

df = pd.DataFrame(data)

st.subheader("📊 Process Overview")
st.table(df)

# Visualization
st.subheader("📈 Process Trends")
time = np.linspace(0, 10, 50)  # minutes
conc_trend = feed_concentration * np.exp(-reaction_rate * time)
temp_trend = np.ones_like(time) * outlet_temp

st.line_chart(pd.DataFrame({
    "Reactor Concentration (mol/L)": conc_trend,
    "Temperature (°C)": temp_trend
}, index=time))

st.success("Adjust parameters from the left sidebar to see how the process changes.")

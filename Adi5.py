import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import plotly.io as pio

st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded")

# Add LinkedIn badge at the top
st.markdown("""
<div style='display: flex; justify-content: flex-start; align-items: center;'>
    <a href="https://www.linkedin.com/in/adityabhatiaquant" target="_blank" style="text-decoration: none; color: white; font-size: 16px;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="margin-right: 8px; vertical-align: middle;"> Aditya Bhatia
    </a>
</div>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.T = time_to_maturity
        self.K = strike
        self.S = current_price
        self.sigma = volatility
        self.r = interest_rate

    def calculate_all(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)

        call = self.S * norm.cdf(d1) - self.K * exp(-self.r * self.T) * norm.cdf(d2)
        put = self.K * exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * sqrt(self.T))

        return call, put, delta_call, delta_put, gamma

# Sidebar inputs
with st.sidebar:
    st.title("\U0001F4CA Black-Scholes Model")
    current_price = st.number_input("Current Asset Price (S)", value=100.0)
    strike = st.number_input("Strike Price (K)", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (T)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate (r)", value=0.05)
    spot_min = st.number_input("Min Spot Price", value=current_price * 0.8)
    spot_max = st.number_input("Max Spot Price", value=current_price * 1.2)
    vol_min = st.slider("Min Volatility", 0.01, 1.0, volatility * 0.5)
    vol_max = st.slider("Max Volatility", 0.01, 1.0, volatility * 1.5)

# Meshgrid setup
spot_range = np.linspace(spot_min, spot_max, 30)
vol_range = np.linspace(vol_min, vol_max, 30)
X, Y = np.meshgrid(spot_range, vol_range)
Z_call = np.zeros_like(X)
Z_put = np.zeros_like(X)
Z_delta = np.zeros_like(X)
Z_gamma = np.zeros_like(X)
Z_theta = np.zeros_like(X)
Z_vega = np.zeros_like(X)
Z_rho = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        bs = BlackScholes(time_to_maturity, strike, X[i, j], Y[i, j], interest_rate)
        call, put, delta_call, delta_put, gamma = bs.calculate_all()

        d1 = (log(X[i, j] / strike) + (interest_rate + 0.5 * Y[i, j]**2) * time_to_maturity) / (Y[i, j] * sqrt(time_to_maturity))
        d2 = d1 - Y[i, j] * sqrt(time_to_maturity)

        vega = X[i, j] * norm.pdf(d1) * sqrt(time_to_maturity)
        theta = (-X[i, j] * norm.pdf(d1) * Y[i, j] / (2 * sqrt(time_to_maturity))) - interest_rate * strike * exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
        rho = strike * time_to_maturity * exp(-interest_rate * time_to_maturity) * norm.cdf(d2)

        Z_call[i, j] = call
        Z_put[i, j] = put
        Z_delta[i, j] = delta_call
        Z_gamma[i, j] = gamma
        Z_theta[i, j] = theta
        Z_vega[i, j] = vega
        Z_rho[i, j] = rho

# Summary metrics
bs_main = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_main, put_main, delta_main, delta_put_main, gamma_main = bs_main.calculate_all()
d1_main = (log(current_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * sqrt(time_to_maturity))
d2_main = d1_main - volatility * sqrt(time_to_maturity)
vega_main = current_price * norm.pdf(d1_main) * sqrt(time_to_maturity)
theta_main = (-current_price * norm.pdf(d1_main) * volatility / (2 * sqrt(time_to_maturity))) - interest_rate * strike * exp(-interest_rate * time_to_maturity) * norm.cdf(d2_main)
rho_main = strike * time_to_maturity * exp(-interest_rate * time_to_maturity) * norm.cdf(d2_main)
st.title("Black-Scholes Option Pricing Summary")
st.table(pd.DataFrame({
    "Metric": ["Call Price", "Put Price", "Call Delta", "Put Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Value": [f"${call_main:.2f}", f"${put_main:.2f}", f"{delta_main:.4f}", f"{delta_put_main:.4f}", f"{gamma_main:.4f}", f"{vega_main:.4f}", f"{theta_main:.4f}", f"{rho_main:.4f}"]
}))

# 3D Surface Plotting
st.title("3D Surfaces for Option Metrics")

def create_3d_surface(z, title, color):
    fig = go.Figure(data=[
        go.Surface(
            z=z, x=X, y=Y, colorscale=color, showscale=True, opacity=0.96,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.4, roughness=0.8, fresnel=0.3),
            lightposition=dict(x=100, y=200, z=0)
        )
    ])
    fig.update_layout(
        title=f"<b>{title}</b>",
        title_font=dict(size=20),
        width=650,
        height=520,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='<b>Spot Price</b>',
            yaxis_title='<b>Volatility</b>',
            zaxis_title=f'<b>{title.split()[0]}</b>',
            xaxis=dict(nticks=6, range=[spot_min, spot_max], backgroundcolor='rgb(10,10,40)'),
            yaxis=dict(nticks=6, range=[vol_min, vol_max], backgroundcolor='rgb(10,10,40)'),
            zaxis=dict(nticks=6, backgroundcolor='rgb(10,10,40)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12)
    )
    return fig

metrics = [
    (Z_call, 'Call Price Surface', 'Blues'),
    (Z_put, 'Put Price Surface', 'YlGnBu'),
    (Z_delta, 'Delta Surface', 'PuBuGn'),
    (Z_gamma, 'Gamma Surface', 'Purples'),
    (Z_vega, 'Vega Surface', 'Blues'),
    (Z_theta, 'Theta Surface', 'OrRd'),
    (Z_rho, 'Rho Surface', 'magma_r')
]

surface_figures = []
for i in range(0, len(metrics), 2):
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_3d_surface(*metrics[i])
        st.plotly_chart(fig1, use_container_width=True)
        surface_figures.append((metrics[i][1], fig1))
    if i + 1 < len(metrics):
        with col2:
            fig2 = create_3d_surface(*metrics[i+1])
            st.plotly_chart(fig2, use_container_width=True)
            surface_figures.append((metrics[i+1][1], fig2))

st.markdown("""
    <div style='margin-top: 10px;'>
        <span style='color:gray;'>💡 Tip: To save your current 3D view, right-click the surface and choose "Save image as..."</span>
    </div>
""", unsafe_allow_html=True)
#pdf report
# PDF Report Generation
# PDF Report Generation
if st.button("📄 Download Full Report as PDF"):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Title Page
        fig_title, ax_title = plt.subplots(figsize=(11, 8.5))
        ax_title.axis('off')
        title_text = ("Black-Scholes Option Pricing Report\n\n"
                      f"Prepared by: Aditya Bhatia\n")
        ax_title.text(0.5, 0.6, title_text, ha='center', va='center', fontsize=16, fontweight='bold')
        ax_title.text(0.5, 0.4, f"Inputs:\nSpot Price: {current_price}, Strike Price: {strike},\nVolatility: {volatility}, Maturity: {time_to_maturity}, Risk-Free Rate: {interest_rate}",
                      ha='center', va='center', fontsize=12)
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close()

        # Description Page
        fig_desc, ax_desc = plt.subplots(figsize=(11, 8.5))
        ax_desc.axis('off')
        description = (
            "Options are financial derivatives that give buyers the right, but not the obligation, to buy or sell an asset at a fixed price.\n\n"
            "\u2022 Call Option: Right to buy the asset\n"
            "\u2022 Put Option: Right to sell the asset\n\n"
            "The Greeks are risk measures that describe how an option's price changes with respect to various factors:\n\n"
            "\u2022 Delta: Sensitivity to underlying asset price\n"
            "        \u03B4 = N(d1)\n"
            "\u2022 Gamma: Sensitivity of delta to underlying price changes\n"
            "        \u0393 = N'(d1) / (S \u03C3 \u221A(T))\n"
            "\u2022 Vega: Sensitivity to volatility\n"
            "        Vega = S N'(d1) \u221A(T)\n"
            "\u2022 Theta: Sensitivity to time decay\n"
            "        \u0398 = - (S N'(d1) \u03C3) / (2 \u221A(T)) - rK e^{-rT} N(d2)\n"
            "\u2022 Rho: Sensitivity to interest rates\n"
            "        \u03C1 = K T e^{-rT} N(d2)"
        )
        ax_desc.text(0.05, 0.95, "Introduction to Options & Greeks", fontsize=16, fontweight='bold', va='top')
        ax_desc.text(0.05, 0.9, description, fontsize=12, va='top')
        pdf.savefig(fig_desc, bbox_inches='tight')
        plt.close()

        # Metric Surface Pages
        for data, title, cmap in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            contour = ax.contourf(X, Y, data, levels=20, cmap=cmap)
            fig.colorbar(contour)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            plt.figtext(0.5, -0.1, f"Strike: {strike}, T: {time_to_maturity}, r: {interest_rate}", wrap=True, ha='center', fontsize=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Summary Table Page
        fig_summary, ax_summary = plt.subplots(figsize=(6, 3))
        ax_summary.axis('off')
        summary_table = pd.DataFrame({
            "Metric": ["Call Price", "Put Price", "Call Delta", "Put Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [f"${call_main:.2f}", f"${put_main:.2f}", f"{delta_main:.4f}", f"{delta_put_main:.4f}",
                      f"{gamma_main:.4f}", f"{vega_main:.4f}", f"{theta_main:.4f}", f"{rho_main:.4f}"]
        })
        table = ax_summary.table(cellText=summary_table.values,
                                 colLabels=summary_table.columns,
                                 cellLoc='center',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax_summary.set_title("Summary of Option Metrics", fontweight='bold')
        pdf.savefig(fig_summary, bbox_inches='tight')
        plt.close()

    buffer.seek(0)
    st.download_button(
        label="\U0001F4E5 Download PDF",
        data=buffer,
        file_name="black_scholes_full_report.pdf",
        mime="application/pdf"
    )

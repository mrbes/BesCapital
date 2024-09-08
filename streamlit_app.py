import streamlit as st
from bitcoin.halving_projection import plot_halving_projection
from bitcoin.one_year_performance import plot_one_year_performance
from bitcoin.bitcoin_power_law import plot_bitcoin_power_law
from bitcoin.monte_carlo_simulations import plot_monte_carlo_simulations
from bitcoin.stress_testing import plot_shock_simulation
from bitcoin.quantile_regression import plot_quantile_regression
from bitcoin.backtesting import plot_backtesting

# Ajouter un en-tête personnalisé
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Analysis Dashboard</h1>
        <p style="font-size:18px;">Explore the various analytical graphs to gain insights into Bitcoin's historical performance and future projections.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Centrer le sélecteur de graphique
chart_type = st.selectbox("Select a chart to display", [
    "Halving Projection",
    "One-Year Performance",
    "Bitcoin Power Law",
    "Monte Carlo Simulations",
    "Shock Simulation",
    "Quantile Regression",
    "Backtesting"
], index=0, help="Choose a graph to view detailed Bitcoin analysis.")

# Utiliser des colonnes pour organiser le contenu
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Afficher le graphique correspondant
    if chart_type == "Halving Projection":
        plot_halving_projection()
        st.markdown("""
            **Halving Projection:** This chart projects Bitcoin prices after the 2024 halving based on past performance. 
            The different colored lines represent the performance from previous halving events, providing a potential future price trajectory.
        """)
    elif chart_type == "One-Year Performance":
        plot_one_year_performance()
        st.markdown("""
            **One-Year Performance:** This graph displays the performance of Bitcoin for one year following each halving event.
            The projection provides insights into potential price behavior post-2024 halving.
        """)
    elif chart_type == "Bitcoin Power Law":
        plot_bitcoin_power_law()
        st.markdown("""
            **Bitcoin Power Law:** This graph illustrates the long-term growth trend of Bitcoin, following a power law. 
            It helps in understanding the potential support and resistance levels based on historical price data.
        """)
    elif chart_type == "Monte Carlo Simulations":
        plot_monte_carlo_simulations()
        st.markdown("""
            **Monte Carlo Simulations:** This chart shows the results of numerous simulated Bitcoin price paths over the next year, 
            using a Monte Carlo approach to model price behavior.
        """)
    elif chart_type == "Shock Simulation":
        plot_shock_simulation()
        st.markdown("""
            **Shock Simulation:** This chart illustrates the potential impact of a mid-term shock on Bitcoin prices, 
            helping to understand how sudden market events could influence future prices.
        """)
    elif chart_type == "Quantile Regression":
        plot_quantile_regression()
        st.markdown("""
            **Quantile Regression:** This graph displays the quantile regression of simulated Bitcoin prices, 
            providing different confidence intervals to assess the range of possible future prices.
        """)
    elif chart_type == "Backtesting":
        plot_backtesting()
        st.markdown("""
            **Backtesting:** This chart compares historical Bitcoin prices with the results of simulations, 
            helping to evaluate the accuracy of predictive models.
        """)

# Ajouter un pied de page
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:16px;">
        © 2024 Analysis Dashboard. All rights reserved. ₿es Capital
    </div>
    """,
    unsafe_allow_html=True
)

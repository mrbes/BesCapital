import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_monte_carlo_simulations():
    # Paramètres de simulation Monte Carlo
    np.random.seed(42)
    num_simulations = 1000
    num_days = 365
    initial_price = 100000  # Prix du Bitcoin au moment du halving 2024
    daily_return_mean = 0.0002
    daily_return_std = 0.02

    # Simulations
    simulations = np.zeros((num_days, num_simulations))
    simulations[0] = initial_price

    for t in range(1, num_days):
        random_shocks = np.random.normal(daily_return_mean, daily_return_std, num_simulations)
        simulations[t] = simulations[t - 1] * (1 + random_shocks)

    # Tracé des simulations
    plt.figure(figsize=(16, 9))
    plt.plot(simulations, color='blue', alpha=0.1)
    plt.yscale('log')
    plt.title('Monte Carlo Simulations of Bitcoin Prices Over 1 Year from 2024 Halving')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.grid(True, which="both", ls="--")
    st.pyplot(plt)

def plot_shock_simulation():
    # Paramètres de simulation avec choc à moyen terme
    np.random.seed(42)
    num_simulations = 1000
    num_days = 365
    initial_price = 100000
    daily_return_mean = 0.0002
    daily_return_std = 0.02

    shock_day = 180
    shock_impact = -0.5  # Choc de -50% au jour 180

    # Simulations
    simulations = np.zeros((num_days, num_simulations))
    simulations[0] = initial_price

    for t in range(1, num_days):
        if t == shock_day:
            simulations[t] = simulations[t - 1] * (1 + shock_impact)
        else:
            random_shocks = np.random.normal(daily_return_mean, daily_return_std, num_simulations)
            simulations[t] = simulations[t - 1] * (1 + random_shocks)

    # Tracé des simulations
    plt.figure(figsize=(16, 9))
    plt.plot(simulations, color='blue', alpha=0.1)
    plt.yscale('log')
    plt.title('Monte Carlo Simulations of Bitcoin Prices with Mid-Term Shock')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.grid(True, which="both", ls="--")
    st.pyplot(plt)

def plot_var_cvar_simulation():
    # Distribution des prix finaux
    final_prices = simulations[-1]
    var_5 = np.percentile(final_prices, 5)
    cvar_5 = final_prices[final_prices <= var_5].mean()

    # Tracé de la distribution des prix finaux
    plt.figure(figsize=(16, 9))
    plt.hist(final_prices, bins=50, color='blue', alpha=0.7)
    plt.axvline(var_5, color='red', linestyle='--', label=f'5% VaR (${var_5:,.2f})')
    plt.axvline(cvar_5, color='orange', linestyle='--', label=f'5% CVaR (${cvar_5:,.2f})')
    plt.title('5% VaR and CVaR of Final Bitcoin Prices')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(plt)

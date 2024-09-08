import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_quantile_regression():
    # Paramètres de simulation Monte Carlo
    np.random.seed(42)
    num_simulations = 1000
    num_days = 365
    initial_price = 100000
    daily_return_mean = 0.0002
    daily_return_std = 0.02

    simulations = np.zeros((num_days, num_simulations))
    simulations[0] = initial_price

    for t in range(1, num_days):
        random_shocks = np.random.normal(daily_return_mean, daily_return_std, num_simulations)
        simulations[t] = simulations[t - 1] * (1 + random_shocks)

    # Calcul des quantiles
    quantiles = np.percentile(simulations, [10, 50, 90], axis=1)

    # Tracé de la régression quantile
    plt.figure(figsize=(16, 9))
    plt.plot(simulations.mean(axis=1), color='blue', label='Mean Trajectory')
    plt.plot(quantiles[0], linestyle='--', color='cyan', label='10.0% Quantile Regression')
    plt.plot(quantiles[1], linestyle='--', color='orange', label='50.0% Quantile Regression')
    plt.plot(quantiles[2], linestyle='--', color='green', label='90.0% Quantile Regression')
    plt.yscale('log')
    plt.title('Quantile Regression of Simulated Bitcoin Prices')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    st.pyplot(plt)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_shock_simulation():
    # Paramètres de simulation
    num_simulations = 1000
    num_days = 365
    initial_price = 100000

    # Générer des simulations de Monte Carlo avec un choc
    simulated_prices = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        simulated_prices[i, 0] = initial_price
        for j in range(1, num_days):
            simulated_prices[i, j] = simulated_prices[i, j-1] * (1 + np.random.normal(0, 0.01))
        
        # Introduire un choc à mi-parcours
        shock_day = num_days // 2
        simulated_prices[i, shock_day:] *= 0.7  # Réduire les prix de 30% à partir du jour de choc

    # Re-tracer les simulations avec le choc
    plt.figure(figsize=(12, 6))
    plt.plot(simulated_prices.T, color='blue', alpha=0.1)
    plt.yscale('log')
    plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])
    plt.title('Monte Carlo Simulations of Bitcoin Prices with Mid-Term Shock')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

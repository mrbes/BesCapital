# bitcoin/backtesting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_backtesting():
    file_path = 'data/uptated_bitcoin.csv'
    df = pd.read_csv(file_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    historical_prices = df['Price']

    # Assurez-vous que les deux colonnes ont la même longueur
    if len(df['Date']) != len(historical_prices):
        min_length = min(len(df['Date']), len(historical_prices))
        df = df.iloc[:min_length]
        historical_prices = historical_prices.iloc[:min_length]

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], historical_prices, color='black', label='Historical Prices')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('Backtesting with Historical Data')
    plt.legend()
    st.pyplot(plt)

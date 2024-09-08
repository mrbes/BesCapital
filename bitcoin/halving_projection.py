# bitcoin/halving_projection.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

plt.style.use('default')

def plot_halving_projection():
    file_path = 'data/uptated_bitcoin.csv'
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])
    halving_dates = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'])

    end_dates_custom = [
        halving_dates[0] + pd.DateOffset(days=1319),
        halving_dates[1] + pd.DateOffset(days=1402),
        halving_dates[2] + pd.DateOffset(days=1440),
        halving_dates[3] + pd.DateOffset(days=1440)
    ]

    scaled_df_list_custom = []
    price_2024_halving = df[df['Date'] == halving_dates[-1]]['Price'].values[0]

    for i in range(len(halving_dates)):
        start_date = halving_dates[i]
        end_date = end_dates_custom[i]
        df_halving = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        start_price = df_halving.iloc[0]['Price']
        df_halving['Scaled Price'] = (df_halving['Price'] / start_price) * price_2024_halving
        df_halving['Days Since Halving'] = (df_halving['Date'] - start_date).dt.days
        scaled_df_list_custom.append(df_halving[['Days Since Halving', 'Scaled Price']])

    aligned_df = scaled_df_list_custom[0].copy()

    performance_custom = {}
    projected_prices_by_halving_custom = {}

    for i in range(len(scaled_df_list_custom) - 1):
        start_price = scaled_df_list_custom[i].iloc[0]['Scaled Price']
        end_price = scaled_df_list_custom[i].iloc[-1]['Scaled Price']
        performance_custom[f'Halving {i+1}'] = (end_price - start_price) / start_price * 100
        projected_prices_by_halving_custom[f'Halving {i+1}'] = price_2024_halving * (1 + performance_custom[f'Halving {i+1}'] / 100)

    halving_colors = {
        'Halving 1': {'performance': 'blue', 'projection': 'blue'},
        'Halving 2': {'performance': 'green', 'projection': 'green'},
        'Halving 3': {'performance': 'red', 'projection': 'red'},
        'Halving 4': {'performance': 'purple', 'projection': 'violet'},
    }
    halving_styles = {
        'Halving 1': {'performance': '-', 'projection': '--'},
        'Halving 2': {'performance': '-', 'projection': '--'},
        'Halving 3': {'performance': '-', 'projection': '--'},
        'Halving 4': {'performance': '-', 'projection': '--'},
    }
    average_color = 'black'
    average_style = '--'

    plt.figure(figsize=(16, 9))

    for i in range(len(scaled_df_list_custom)):
        halving_label = f'Halving {i+1}'
        plt.plot(scaled_df_list_custom[i]['Days Since Halving'],
                 scaled_df_list_custom[i]['Scaled Price'],
                 label=f'{halving_label} Performance on 2024 Price',
                 color=halving_colors[halving_label]['performance'],
                 linestyle=halving_styles[halving_label]['performance'],
                 linewidth=2 if halving_label == 'Halving 4' else 1)

    plt.plot(aligned_df['Days Since Halving'], aligned_df['Scaled Price'], color=average_color, linestyle=average_style, linewidth=1,
             label='Average Performance on 2024 Price')

    plt.axhline(y=price_2024_halving, color='black', linestyle='--', label=f'2024 Halving Price (${price_2024_halving:,.2f})')

    for i, (key, value) in enumerate(projected_prices_by_halving_custom.items()):
        plt.axhline(y=value, color=halving_colors[f'Halving {i+1}']['projection'],
                    linestyle=halving_styles[f'Halving {i+1}']['projection'],
                    label=f'{key} Projected Price (${value:,.2f})')

    plt.yscale('log')
    plt.yticks([10**i for i in range(4, int(np.log10(max(aligned_df['Scaled Price']))+2))],
               [f'${10**i:,.0f}' for i in range(4, int(np.log10(max(aligned_df['Scaled Price']))+2))])

    plt.annotate('@Besbtc', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=20, color='gray', ha='center')

    plt.title('Projected Bitcoin Price After 2024 Halving Based on Past Performances')
    plt.xlabel('Days Since Halving')
    plt.ylabel('Projected Bitcoin Price (USD)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    st.pyplot(plt)

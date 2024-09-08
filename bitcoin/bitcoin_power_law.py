import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import streamlit as st

# Fonction pour tracer le graphique du Bitcoin Power Law & Bandes
def plot_bitcoin_power_law():
    # Appliquer le thème sombre seulement pour ce graphique
    with plt.style.context('dark_background'):
        file_path = 'data/uptated_bitcoin.csv'  # Assurez-vous que le chemin est correct
        bitcoin_data = pd.read_csv(file_path, parse_dates=['Date'])

        # Convertir la colonne 'Date' en datetime
        bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])

        # Calculer les jours depuis le bloc Genesis (03-01-2009)
        genesis_block_date = datetime(2009, 1, 3)
        bitcoin_data['Jours_depuis_Genesis'] = (bitcoin_data['Date'] - genesis_block_date).dt.days

        # Définir la loi de puissance
        def power_law_trend(days):
            A = 10**-17
            n = 5.8
            return A * (days ** n)

        # Prolonger les prédictions jusqu'en 2045
        end_date = datetime(2045, 1, 1)
        extended_dates = pd.date_range(start=bitcoin_data['Date'].min(), end=end_date, freq='D')
        days_since_genesis_extended = (extended_dates - genesis_block_date).days

        # Calculer la tendance pour la période prolongée
        trend_extended = power_law_trend(days_since_genesis_extended)

        # Calculer les bandes de tolérance pour la période prolongée
        bands_extended = {
            '-50% - -25%': (trend_extended * 0.5, trend_extended * 0.75),
            '-25% - 0%': (trend_extended * 0.75, trend_extended),
            '0% - +100%': (trend_extended, trend_extended * 2),
            '+100% - +200%': (trend_extended * 2, trend_extended * 3),
            '+200% - +300%': (trend_extended * 3, trend_extended * 4),
            '+300% - +400%': (trend_extended * 4, trend_extended * 5),
            '+400% - +500%': (trend_extended * 5, trend_extended * 6),
        }

        # Créer le tracé pour la période prolongée
        plt.figure(figsize=(16, 9))
        plt.plot(bitcoin_data['Date'], bitcoin_data['Price'], color='white', label='Price')
        plt.plot(extended_dates, trend_extended, 'cyan', linestyle='-', label='Power Law')

        # Ajouter les bandes de tolérance prolongées
        colors = ['darkgreen', 'green', 'yellow', 'orange', 'red', 'brown', 'darkred']
        for i, (label, (lower, upper)) in enumerate(bands_extended.items()):
            plt.fill_between(extended_dates, lower, upper, color=colors[i], alpha=0.5, label=label)

        # Ajouter les halvings
        halving_dates = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20']
        for halving in halving_dates:
            plt.axvline(pd.to_datetime(halving), color='magenta', linestyle=':', label='Halving' if halving == halving_dates[0] else "")

        # Configurer les axes
        plt.yscale('log')
        plt.ylim(0.1, 200000)
        plt.yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], ['$1', '$10', '$100', '$1k', '$10k', '$100k','$1M', '$10M'])

        # Custom date ticks
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xlim(bitcoin_data['Date'].min(), end_date)
        plt.xlabel('')
        plt.ylabel('$')
        plt.legend(loc='best')
        plt.title('Bitcoin Power Law & Bandes')

        # Supprimer toutes les grilles
        plt.grid(False)

        # Ajouter un texte pour marquer le projet
        plt.text(0.5, 0.5, '@Besbtc', fontsize=20, color='white', ha='center', va='center', alpha=0.5, transform=plt.gcf().transFigure)

        # Afficher le graphique dans Streamlit
        st.pyplot(plt)

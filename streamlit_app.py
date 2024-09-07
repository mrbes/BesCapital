import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour créer le graphique
def plot_halving_projection():
    # Recharger les données depuis le fichier CSV
    file_path = 'data/uptated_bitcoin.csv'  # Ajustez le chemin selon l'endroit où se trouve votre fichier
    df = pd.read_csv(file_path)

    # Convertir la colonne 'Date' au format datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Définir les dates des halvings
    halving_dates = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'])

    # Utiliser le nombre exact de jours entre chaque halving pour les dates de fin personnalisées
    end_dates_custom = [
        halving_dates[0] + pd.DateOffset(days=1319),
        halving_dates[1] + pd.DateOffset(days=1402),
        halving_dates[2] + pd.DateOffset(days=1440),
        halving_dates[3] + pd.DateOffset(days=1440)
    ]

    # Préparer les données pour chaque halving avec les jours ajustés
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

    # Initialiser aligned_df comme une copie du premier DF
    aligned_df = scaled_df_list_custom[0].copy()

    # Recalculer les performances sur la période complète et les "Halving Projected Prices"
    performance_custom = {}
    projected_prices_by_halving_custom = {}

    for i in range(len(scaled_df_list_custom) - 1):  # Exclude Halving 4 projection
        start_price = scaled_df_list_custom[i].iloc[0]['Scaled Price']
        end_price = scaled_df_list_custom[i].iloc[-1]['Scaled Price']
        performance_custom[f'Halving {i+1}'] = (end_price - start_price) / start_price * 100
        projected_prices_by_halving_custom[f'Halving {i+1}'] = price_2024_halving * (1 + performance_custom[f'Halving {i+1}'] / 100)

    # Assurez-vous que ces variables sont correctement définies
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

    # Créer le graphique mis à jour avec les "Halving Projected Prices" pour la période complète
    plt.figure(figsize=(16, 9))

    for i in range(len(scaled_df_list_custom)):
        halving_label = f'Halving {i+1}'

        # Tracer la courbe de performance avec épaisseur personnalisée pour le Halving 4
        plt.plot(scaled_df_list_custom[i]['Days Since Halving'],
                 scaled_df_list_custom[i]['Scaled Price'],
                 label=f'{halving_label} Performance on 2024 Price',
                 color=halving_colors[halving_label]['performance'],
                 linestyle=halving_styles[halving_label]['performance'],
                 linewidth=2 if halving_label == 'Halving 4' else 1)

    # Ajouter la courbe de performance moyenne avec épaisseur à 1
    plt.plot(aligned_df['Days Since Halving'], aligned_df['Scaled Price'], color=average_color, linestyle=average_style, linewidth=1,
             label='Average Performance on 2024 Price')

    # Ajouter la ligne horizontale pour le prix de départ en 2024
    plt.axhline(y=price_2024_halving, color='black', linestyle='--', label=f'2024 Halving Price (${price_2024_halving:,.2f})')

    # Ajouter les prix projetés sauf pour le Halving 4
    for i, (key, value) in enumerate(projected_prices_by_halving_custom.items()):
        plt.axhline(y=value, color=halving_colors[f'Halving {i+1}']['projection'],
                    linestyle=halving_styles[f'Halving {i+1}']['projection'],
                    label=f'{key} Projected Price (${value:,.2f})')

    # Mettre l'axe y en échelle logarithmique en commençant à $10,000
    plt.yscale('log')

    # Ajouter des étiquettes numériques à l'axe y, en commençant à $10,000
    plt.yticks([10**i for i in range(4, int(np.log10(max(aligned_df['Scaled Price']))+2))],
               [f'${10**i:,.0f}' for i in range(4, int(np.log10(max(aligned_df['Scaled Price']))+2))])

    # Ajouter une annotation au milieu du graphique
    plt.annotate('@Besbtc', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=20, color='gray', ha='center')

    plt.title('Projected Bitcoin Price After 2024 Halving Based on Past Performances')
    plt.xlabel('Days Since Halving')
    plt.ylabel('Projected Bitcoin Price (USD)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

# Titre de l'application
st.title("Bitcoin Halving Performance Dashboard")

# Appel de la fonction pour afficher le graphique
plot_halving_projection()

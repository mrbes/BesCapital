import streamlit as st
<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour le premier graphique (Projection après halving)
def plot_halving_projection():
    file_path = 'data/uptated_bitcoin.csv'  # Assurez-vous que le chemin est correct
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

# Fonction pour le deuxième graphique (Performance sur un an)
def plot_second_graph():
    file_path = 'data/uptated_bitcoin.csv'
    df = pd.read_csv(file_path)

    # Convertir la colonne 'Date' au format datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Définir les dates des halvings
    halving_dates = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20']
    halving_dates = pd.to_datetime(halving_dates)

    # Calculer 365 jours après chaque halving
    end_dates = halving_dates + pd.DateOffset(days=365)

    # Préparer les données pour chaque halving
    scaled_df_list = []
    price_2024_halving = df[df['Date'] == halving_dates[-1]]['Price'].values[0]

    for i in range(len(halving_dates)):
        start_date = halving_dates[i]
        end_date = end_dates[i]
        df_halving = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        start_price = df_halving.iloc[0]['Price']
        df_halving['Scaled Price'] = (df_halving['Price'] / start_price) * price_2024_halving

        df_halving['Days Since Halving'] = (df_halving['Date'] - start_date).dt.days
        scaled_df_list.append(df_halving[['Days Since Halving', 'Scaled Price']])

    # Calculer la performance de chaque halving
    performance = {}
    for i in range(len(halving_dates)-1):
        start_price = scaled_df_list[i].iloc[0]['Scaled Price']
        end_price = scaled_df_list[i].iloc[-1]['Scaled Price']
        performance[f'Halving {i+1}'] = (end_price - start_price) / start_price * 100

    # Calculer les prix projetés
    projected_prices_by_halving = {}
    for i in range(len(performance)):
        projected_prices_by_halving[f'Halving {i+1}'] = price_2024_halving * (1 + performance[f'Halving {i+1}'] / 100)

    # Calculer la performance moyenne et le prix projeté pour 2024
    average_performance = sum(performance.values()) / len(performance)
    average_projected_price_2024 = price_2024_halving * (1 + average_performance / 100)

    # Paramètres personnalisés pour chaque halving et sa projection
    halving_colors = {
        'Halving 1': {'performance': 'blue', 'projection': 'blue'},
        'Halving 2': {'performance': 'green', 'projection': 'green'},
        'Halving 3': {'performance': 'red', 'projection': 'red'},
        'Halving 4': {'performance': 'black', 'projection': 'orange'}
    }

    halving_styles = {
        'Halving 1': {'performance': '-', 'projection': '--'},
        'Halving 2': {'performance': '-', 'projection': '--'},
        'Halving 3': {'performance': '-', 'projection': '--'},
        'Halving 4': {'performance': '-', 'projection': '--'}
    }

    # Couleur et style de la ligne projetée moyenne
    average_color = 'purple'
    average_style = '-'

    # Création du graphique avec les paramètres personnalisés
    plt.figure(figsize=(16, 9))

    for i in range(len(halving_dates)):
        halving_label = f'Halving {i+1}'

        # Tracer la courbe de performance
        plt.plot(scaled_df_list[i]['Days Since Halving'],
                 scaled_df_list[i]['Scaled Price'],
                 label=f'{halving_label} Performance on 2024 Price',
                 color=halving_colors[halving_label]['performance'],
                 linestyle=halving_styles[halving_label]['performance'])

    # Ajouter le point de départ de 2024 comme ligne horizontale
    plt.axhline(y=price_2024_halving, color='black', linestyle='--', label=f'2024 Halving Price (${price_2024_halving:,.2f})')

    # Ajouter les prix projetés avec des paramètres personnalisés
    for i, (key, value) in enumerate(projected_prices_by_halving.items()):
        halving_label = f'Halving {i+1}'
        plt.axhline(y=value, color=halving_colors[halving_label]['projection'],
                    linestyle=halving_styles[halving_label]['projection'],
                    label=f'{key} Projected Price (${value:,.2f})')

    # Ajouter le prix projeté moyen
    plt.axhline(y=average_projected_price_2024, color=average_color, linestyle=average_style, linewidth=2,
                label=f'Average Projected Price (${average_projected_price_2024:,.2f})')

    # Mettre l'axe y en échelle logarithmique
    plt.yscale('log')

    # Ajouter des étiquettes numériques à l'axe y
    plt.yticks([10**i for i in range(4, int(np.log10(average_projected_price_2024))+2)],
               [f'${10**i:,.0f}' for i in range(4, int(np.log10(average_projected_price_2024))+2)])

    # Ajouter une annotation au milieu du graphique
    plt.annotate('@Besbtc', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=20, color='gray', ha='center')

    plt.title('Projected Bitcoin Price After 2024 Halving Based on Past Performances (Logarithmic Scale)')
    plt.xlabel('Days Since Halving')
    plt.ylabel('Projected Bitcoin Price (USD)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

# Titre de l'application
st.title("Bitcoin Halving Performance Dashboard")

# Sélecteur de graphique
chart_type = st.selectbox("Select a chart to display", ["Halving Projection", "One-Year Performance"])

# Afficher le graphique correspondant
if chart_type == "Halving Projection":
    plot_halving_projection()
elif chart_type == "One-Year Performance":
    plot_second_graph()
=======
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
>>>>>>> ed5e4b7 (Initial commit avec les nouveaux scripts et la mise à jour de l'application Streamlit)

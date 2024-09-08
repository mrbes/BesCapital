# Cette cellule est nécessaire uniquement si les packages ne sont pas encore installés dans votre environnement Colab
!pip install pandas matplotlib statsmodels


pip install arch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import matplotlib.image as mpimg

# Charger les données Bitcoin depuis le fichier CSV
file_path = '/content/uptated_bitcoin.csv'
bitcoin_data = pd.read_csv(file_path)

# Convertir la colonne 'Date' en datetime
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])

# Calculer les jours depuis le bloc Genesis (03-01-2009)
genesis_block_date = datetime.datetime(2009, 1, 3)
bitcoin_data['Jours_depuis_Genesis'] = (bitcoin_data['Date'] - genesis_block_date).dt.days

# Calculer le logarithme du prix
bitcoin_data['Log_Price'] = np.log(bitcoin_data['Price'])

# Calculer le logarithme du temps (en jours depuis le Genesis Block)
bitcoin_data['Log_Time'] = np.log(bitcoin_data['Jours_depuis_Genesis'])


# Vérifier les valeurs manquantes ou infinies
print("Valeurs manquantes dans Log_Price :", bitcoin_data['Log_Price'].isnull().sum())
print("Valeurs infinies dans Log_Price :", np.isinf(bitcoin_data['Log_Price']).sum())

print("Valeurs manquantes dans Log_Time :", bitcoin_data['Log_Time'].isnull().sum())
print("Valeurs infinies dans Log_Time :", np.isinf(bitcoin_data['Log_Time']).sum())

# Supprimer les lignes avec des valeurs manquantes ou infinies
bitcoin_data_clean = bitcoin_data[~bitcoin_data['Log_Price'].isnull() & ~np.isinf(bitcoin_data['Log_Price'])]
bitcoin_data_clean = bitcoin_data_clean[~bitcoin_data_clean['Log_Time'].isnull() & ~np.isinf(bitcoin_data_clean['Log_Time'])]

# Régression quantile avec le solveur 'highs'
quantile = 0.05  # 5e percentile pour la ligne de support
model = QuantileRegressor(quantile=quantile, alpha=0, solver='highs').fit(
    bitcoin_data_clean[['Log_Time']], bitcoin_data_clean['Log_Price'])
bitcoin_data_clean['Support'] = model.predict(bitcoin_data_clean[['Log_Time']])


# Utiliser un style sombre global
plt.style.use('dark_background')

# Tracer les données et la régression
plt.figure(figsize=(10, 6))
plt.plot(bitcoin_data_clean['Log_Time'], bitcoin_data_clean['Log_Price'], label='Log of Price', color='white', linewidth=1)
plt.plot(bitcoin_data_clean['Log_Time'], bitcoin_data_clean['Support'], label=f'{quantile} Quantile Regression', color='red', linestyle='--', linewidth=1)
#plt.xlabel('Log of Time')
#plt.ylabel('Log of Price')
plt.title('Bitcoin Power Law')

# Améliorer l'axe des prix
price_ticks = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]  # En dollars
plt.yticks(np.log(price_ticks), ['$0.1', '$1', '$10', '$100', '$1k', '$10k', '$100k', '$1M'])

# Améliorer l'axe des dates
time_ticks = [np.log((datetime.datetime(year, 12, 31) - genesis_block_date).days) for year in range(2011, 2045, 2)]
time_labels = [f'{year}' for year in range(2011, 2045, 2)]
plt.xticks(time_ticks, time_labels, rotation=90)

#plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1)
plt.show()


# Utiliser un style sombre global
plt.style.use('dark_background')

# Définir les quantiles pour les régressions basse et haute
low_quantile = 0.03  # Quantile pour la ligne de support (3e percentile)
high_quantile = 0.97  # Quantile pour la ligne de résistance (97e percentile)

# Modèle pour la régression basse
low_model = QuantileRegressor(quantile=low_quantile, alpha=0, solver='highs').fit(
    bitcoin_data_clean[['Log_Time']], bitcoin_data_clean['Log_Price'])
bitcoin_data_clean['Support'] = low_model.predict(bitcoin_data_clean[['Log_Time']])

# Modèle pour la régression haute
high_model = QuantileRegressor(quantile=high_quantile, alpha=0, solver='highs').fit(
    bitcoin_data_clean[['Log_Time']], bitcoin_data_clean['Log_Price'])
bitcoin_data_clean['Resistance'] = high_model.predict(bitcoin_data_clean[['Log_Time']])

# Tracer les données et les régressions
plt.figure(figsize=(10, 6))

# Ligne de prix
plt.plot(bitcoin_data_clean['Log_Time'], bitcoin_data_clean['Log_Price'], label='Log of Price', color='white', linewidth=1)

# Lignes de régressions
plt.plot(bitcoin_data_clean['Log_Time'], bitcoin_data_clean['Support'], label=f'{low_quantile} Quantile Regression', color='red', linestyle='--', linewidth=1)
plt.plot(bitcoin_data_clean['Log_Time'], bitcoin_data_clean['Resistance'], label=f'{high_quantile} Quantile Regression', color='cyan', linestyle='--', linewidth=1)

# Titre du graphique
plt.title('Bitcoin Power Law Support and Resistance')

# Améliorer l'axe des prix
price_ticks = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]  # En dollars
plt.yticks(np.log(price_ticks), ['$0.1', '$1', '$10', '$100', '$1k', '$10k', '$100k', '$1M'])

# Améliorer l'axe des dates
time_ticks = [np.log((datetime.datetime(year, 12, 31) - genesis_block_date).days) for year in range(2011, 2045, 2)]
time_labels = [f'{year}' for year in range(2011, 2045, 2)]
plt.xticks(time_ticks, time_labels, rotation=90)

# Ajouter la grille
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.05)

#plt.legend()

# Afficher le graphique
plt.show()



# Dates exactes des halvings
halving_dates = {
    1: '2012-11-28',
    2: '2016-07-09',
    3: '2020-05-11',
    4: '2024-04-20'
}

# Convertir les dates en format datetime
bitcoin_data_clean['Date'] = pd.to_datetime(bitcoin_data_clean['Date'])
halving_dates = {k: pd.to_datetime(v) for k, v in halving_dates.items()}

# Filtrer les données pour chaque halving
performances = []
for i in range(1, 4):
    start_date = halving_dates[i]
    end_date = start_date + pd.Timedelta(days=365)
    halving_data = bitcoin_data_clean[(bitcoin_data_clean['Date'] >= start_date) & (bitcoin_data_clean['Date'] <= end_date)]
    if len(halving_data) > 0:  # Vérifie qu'il y a des données après le halving
        performances.append(halving_data)

# Vérification si nous avons assez de périodes pour calculer les rendements
if len(performances) == 0:
    raise ValueError("Pas de données suffisantes pour exécuter les simulations Monte Carlo.")

# Calcul des rendements log-normalisés pour les périodes disponibles
log_returns = []
for data in performances:
    log_return = np.diff(np.log(data['Price'].values))
    log_returns.extend(log_return)

log_returns = np.array(log_returns)
mean_return = np.mean(log_returns)
std_dev_return = np.std(log_returns)

# Simulation Monte Carlo pour 2024
num_simulations = 10000
num_days = 365
last_price = bitcoin_data_clean[bitcoin_data_clean['Date'] == halving_dates[4]]['Price'].values[0]

simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = last_price

for i in range(1, num_days):
    random_shock = np.random.normal(mean_return, std_dev_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * np.exp(random_shock)

# Tracer les simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices.T, color='blue', alpha=0.1)

# Réglage manuel des étiquettes de l'axe des prix pour éviter la notation scientifique
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])

plt.title('Monte Carlo Simulations of Bitcoin Prices Over 1 Year from 2024 Halving')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()



# Dates exactes des halvings
halving_dates = {
    1: '2012-11-28',
    2: '2016-07-09',
    3: '2020-05-11',
    4: '2024-04-20'
}

# Convertir les dates en format datetime
bitcoin_data_clean['Date'] = pd.to_datetime(bitcoin_data_clean['Date'])
halving_dates = {k: pd.to_datetime(v) for k, v in halving_dates.items()}

# Filtrer les données pour chaque halving
performances = []
for i in range(1, 4):
    start_date = halving_dates[i]
    end_date = start_date + pd.Timedelta(days=365)
    halving_data = bitcoin_data_clean[(bitcoin_data_clean['Date'] >= start_date) & (bitcoin_data_clean['Date'] <= end_date)]
    if len(halving_data) > 0:  # Vérifie qu'il y a des données après le halving
        performances.append(halving_data)

# Vérification si nous avons assez de périodes pour calculer les rendements
if len(performances) == 0:
    raise ValueError("Pas de données suffisantes pour exécuter les simulations Monte Carlo.")

# Facteur de contrôle de la volatilité
volatility_factor = 0.5  # 1.0 pour garder la volatilité actuelle, <1 pour réduire, >1 pour augmenter

# Calcul des rendements log-normalisés pour les périodes disponibles
log_returns = []
for data in performances:
    log_return = np.diff(np.log(data['Price'].values))
    log_returns.extend(log_return)

log_returns = np.array(log_returns)
mean_return = np.mean(log_returns)
std_dev_return = np.std(log_returns) * volatility_factor  # Appliquer le facteur de contrôle de la volatilité

# Simulation Monte Carlo pour 2024
num_simulations = 100
num_days = 365
last_price = bitcoin_data_clean[bitcoin_data_clean['Date'] == halving_dates[4]]['Price'].values[0]

simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = last_price

for i in range(1, num_days):
    random_shock = np.random.normal(mean_return, std_dev_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * np.exp(random_shock)

# Tracer les simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices.T, color='blue', alpha=0.1)

# Réglage manuel des étiquettes de l'axe des prix pour éviter la notation scientifique
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])

plt.title('Monte Carlo Simulations of Bitcoin Prices Over 1 Year from 2024 Halving')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()




# Paramètres de la simulation Monte Carlo
num_simulations = 1000  # Nombre de simulations Monte Carlo
num_days = 365  # Durée de la simulation (1 an)

# Calcul des rendements log-normalisés et des statistiques associées
log_returns = np.diff(np.log(bitcoin_data_clean['Price'].values))
mean_return = np.mean(log_returns)
std_dev_return = np.std(log_returns)

# Dernier prix connu
last_price = bitcoin_data_clean['Price'].values[-1]

# Initialiser les simulations
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = last_price

# Exécution des simulations Monte Carlo
for i in range(1, num_days):
    random_shock = np.random.normal(mean_return, std_dev_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * np.exp(random_shock)

# Tracer les simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices.T, color='blue', alpha=0.1)

# Réglage manuel des étiquettes de l'axe des prix pour éviter la notation scientifique
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])



plt.title('Monte Carlo Simulations of Bitcoin Prices Over 1 Year')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True, linestyle='--', linewidth=0.5)


plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Supposons que `final_day_prices` ait été calculé précédemment

# Définir les seuils de prix pour les probabilités
price_thresholds = [50_000, 100_000, 200_000, 500_000]

# Calcul des probabilités que le prix dépasse chaque seuil d'ici la fin de l'année
probabilities = {}
for threshold in price_thresholds:
    prob = np.mean(final_day_prices > threshold)
    probabilities[threshold] = prob

# Calcul de l'intervalle de confiance à 95% des prix
confidence_interval = np.percentile(final_day_prices, [2.5, 97.5])

# Tracer la distribution des prix finaux
plt.figure(figsize=(12, 6))
plt.hist(final_day_prices, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

# Marquer l'intervalle de confiance
plt.axvline(confidence_interval[0], color='green', linestyle='-', linewidth=2, label=f"2.5% Quantile ({confidence_interval[0]:,.0f} $)")
plt.axvline(confidence_interval[1], color='purple', linestyle='-', linewidth=2, label=f"97.5% Quantile ({confidence_interval[1]:,.0f} $)")

# Formater l'axe des prix avec un espacement plus grand, généré automatiquement
ticks = np.arange(0, final_day_prices.max(), step=500_000)
tick_labels = [f'{int(tick/1e6)}M' if tick >= 1e6 else f'{int(tick/1e3)}K' for tick in ticks]
plt.xticks(ticks, tick_labels)

plt.title('Distribution des Prix du Bitcoin dans 1 An (Simulations Monte Carlo)')
plt.xlabel('Prix du Bitcoin ($)')
plt.ylabel('Fréquence')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()




# Supposons que `final_day_prices` soit déjà calculé

# Définir les seuils de prix pour les probabilités
price_thresholds = [50_000, 100_000, 200_000, 500_000, 1_000_000]

# Calcul des probabilités que le prix dépasse chaque seuil
probabilities = {}
for threshold in price_thresholds:
    prob = np.mean(final_day_prices > threshold)
    probabilities[threshold] = prob
    print(f"Probabilité que le Bitcoin dépasse {threshold:,} $ d'ici 1 an: {prob*100:.2f}%")

# Calcul des scénarios optimistes et pessimistes
top_1_percentile = np.percentile(final_day_prices, 99)
bottom_1_percentile = np.percentile(final_day_prices, 1)

print(f"\nScénario optimiste (99ème percentile) : Prix du Bitcoin supérieur à {top_1_percentile:,.0f} $")
print(f"Scénario pessimiste (1er percentile) : Prix du Bitcoin inférieur à {bottom_1_percentile:,.0f} $")



# Tracer la distribution des prix avec une courbe de densité
plt.figure(figsize=(12, 6))

# Histogramme
sns.histplot(final_day_prices, bins=50, kde=True, color='skyblue', edgecolor='black')

# Marquer les scénarios optimistes et pessimistes
plt.axvline(top_1_percentile, color='orange', linestyle='-', linewidth=2, label=f"99ème percentile ({top_1_percentile:,.0f} $)")
plt.axvline(bottom_1_percentile, color='red', linestyle='-', linewidth=2, label=f"1er percentile ({bottom_1_percentile:,.0f} $)")

# Marquer l'intervalle de confiance à 95%
plt.axvline(confidence_interval[0], color='green', linestyle='-', linewidth=2, label=f"2.5% Quantile ({confidence_interval[0]:,.0f} $)")
plt.axvline(confidence_interval[1], color='purple', linestyle='-', linewidth=2, label=f"97.5% Quantile ({confidence_interval[1]:,.0f} $)")

# Formater l'axe des prix avec des unités
plt.xticks(np.arange(0, final_day_prices.max(), step=500_000),
           [f'{int(tick/1e6)}M' if tick >= 1e6 else f'{int(tick/1e3)}K' for tick in np.arange(0, final_day_prices.max(), step=500_000)])

plt.title('Distribution des Prix du Bitcoin dans 1 An avec Scénarios Optimistes et Pessimistes')
plt.xlabel('Prix du Bitcoin ($)')
plt.ylabel('Fréquence')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()



# Supposons que `final_day_prices` soit déjà calculé

# Créer la fonction de répartition empirique (ECDF)
ecdf = ECDF(final_day_prices)

# Tracer la courbe de la probabilité cumulée
plt.figure(figsize=(12, 6))
plt.plot(ecdf.x, ecdf.y, marker='.', linestyle='none')

# Ajuster l'échelle de l'axe des prix à une échelle logarithmique
plt.xscale('log')

# Ajuster les ticks et étiquettes pour l'axe des prix
ticks = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
tick_labels = [f'{int(tick):,}' for tick in ticks]
plt.xticks(ticks, tick_labels)

plt.xlabel('Prix du Bitcoin ($)')
plt.ylabel('Probabilité Cumulée')
plt.title('Fonction de Répartition Empirique (ECDF) des Prix Simulés du Bitcoin')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


# Facteur de contrôle de la volatilité
volatility_factor = 1.0  # 1.0 pour garder la volatilité actuelle, <1 pour réduire, >1 pour augmenter

# Facteur de contrôle de la moyenne des rendements
mean_factor = 0.5  # 1.0 pour garder la moyenne actuelle, <1 pour réduire (scénario pessimiste), >1 pour augmenter (scénario optimiste)

# Calcul des rendements log-normalisés pour les périodes disponibles
log_returns = []
for data in performances:
    log_return = np.diff(np.log(data['Price'].values))
    log_returns.extend(log_return)

log_returns = np.array(log_returns)
mean_return = np.mean(log_returns) * mean_factor  # Appliquer le facteur de contrôle de la moyenne
std_dev_return = np.std(log_returns) * volatility_factor  # Appliquer le facteur de contrôle de la volatilité

# Simulation Monte Carlo pour 2024
num_simulations = 1000
num_days = 365
last_price = bitcoin_data_clean[bitcoin_data_clean['Date'] == halving_dates[4]]['Price'].values[0]

simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = last_price

for i in range(1, num_days):
    random_shock = np.random.normal(mean_return, std_dev_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * np.exp(random_shock)

# Tracer les simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices.T, color='blue', alpha=0.1)

# Réglage manuel des étiquettes de l'axe des prix pour éviter la notation scientifique
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])

plt.title('Monte Carlo Simulations of Bitcoin Prices Over 1 Year from 2024 Halving')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


# Facteur de contrôle de la volatilité
volatility_factor = 1.0  # 1.0 pour garder la volatilité actuelle, <1 pour réduire, >1 pour augmenter

# Facteur de contrôle de la moyenne des rendements
mean_factor = 0.5  # 1.0 pour garder la moyenne actuelle, <1 pour réduire (scénario pessimiste), >1 pour augmenter (scénario optimiste)

# Calcul des rendements log-normalisés pour les périodes disponibles
log_returns = []
for data in performances:
    log_return = np.diff(np.log(data['Price'].values))
    log_returns.extend(log_return)

log_returns = np.array(log_returns)
mean_return = np.mean(log_returns) * mean_factor  # Appliquer le facteur de contrôle de la moyenne
std_dev_return = np.std(log_returns) * volatility_factor  # Appliquer le facteur de contrôle de la volatilité

# Simulation Monte Carlo pour 2024
num_simulations = 1000
num_days = 365
last_price = bitcoin_data_clean[bitcoin_data_clean['Date'] == halving_dates[4]]['Price'].values[0]

simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = last_price

for i in range(1, num_days):
    random_shock = np.random.normal(mean_return, std_dev_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * np.exp(random_shock)

# Calcul du prix moyen à chaque jour
mean_prices = np.mean(simulated_prices, axis=0)

# Tracer les simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_prices.T, color='blue', alpha=0.1)

# Tracer la trajectoire moyenne en noir
plt.plot(mean_prices, color='black', linewidth=2, label='Mean Price')

# Réglage manuel des étiquettes de l'axe des prix pour éviter la notation scientifique
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])

plt.title('Monte Carlo Simulations of Bitcoin Prices Over 1 Year from 2024 Halving')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()


final_prices = simulated_prices[:, -1]

plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, color='blue', alpha=0.7)

# Mettre à jour les étiquettes de l'axe X en format numérique
plt.xticks(plt.xticks()[0], [f'{int(x):,}' for x in plt.xticks()[0]])

plt.xlabel('Final Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Final Bitcoin Prices After 1 Year')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Vos données
thresholds = [50000, 100000, 150000, 200000, 250000, 500000, 1000000]
probabilities = [np.sum(final_prices > threshold) / num_simulations for threshold in thresholds]

# Création du graphique à barres
plt.figure(figsize=(8, 6))
plt.bar(thresholds, probabilities, color='blue', alpha=0.7)

# Ajouter les points aux endroits où les pourcentages sont affichés
plt.scatter(thresholds, probabilities, color='red', zorder=5)  # Marqueurs rouges pour les points

# Ajouter les valeurs exactes au-dessus des barres
for i in range(len(thresholds)):
    plt.text(thresholds[i], probabilities[i] + 0.01, f'{probabilities[i] * 100:.2f}%', ha='center', fontsize=12)

# Tracer une ligne de régression sur les points
X = np.array(thresholds).reshape(-1, 1)
y = np.array(probabilities)
reg = LinearRegression().fit(X, y)
reg_line = reg.predict(X)

plt.plot(thresholds, reg_line, color='green', linestyle='--', label='Linear Regression')  # Ligne de régression en vert

# Limiter l'axe des Y entre 0 et 100%
plt.ylim(0, 1)

# Mettre à jour les étiquettes de l'axe X en format numérique avec une rotation de 90 degrés
plt.xticks(thresholds, [f'{x:,.0f}' for x in thresholds], rotation=90)

plt.xlabel('Price Threshold ($)')
plt.ylabel('Probability')
plt.title('Probability of Bitcoin Price Exceeding Different Thresholds After 1 Year')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()


mean_trajectory = np.mean(simulated_prices, axis=0)
upper_bound = np.percentile(simulated_prices, 95, axis=0)
lower_bound = np.percentile(simulated_prices, 5, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(mean_trajectory, color='blue', label='Mean Trajectory')
plt.fill_between(range(num_days), lower_bound, upper_bound, color='blue', alpha=0.3, label='90% Confidence Interval')
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Average Bitcoin Price Trajectory with 90% Confidence Interval')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()


daily_volatility = np.std(simulated_prices, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(daily_volatility, color='blue', label='Daily Volatility')
plt.xlabel('Days')
plt.ylabel('Volatility ($)')
plt.title('Daily Volatility of Bitcoin Prices Over 1 Year')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


plt.figure(figsize=(12, 6))
plt.imshow(np.log(simulated_prices), aspect='auto', cmap='gist_rainbow_r', interpolation='nearest')
plt.colorbar(label='Log(Price)')
plt.title('Heatmap of Simulated Bitcoin Prices Over 1 Year')
plt.xlabel('Days')
plt.ylabel('Simulation Number')
plt.show()


all_daily_returns = np.diff(np.log(simulated_prices), axis=1).flatten()

plt.figure(figsize=(10, 6))
plt.hist(all_daily_returns, bins=100, color='blue', alpha=0.7, density=True)
plt.xlabel('Daily Log Returns')
plt.ylabel('Density')
plt.title('Distribution of Simulated Daily Log Returns')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


mean_trajectory = np.mean(simulated_prices, axis=0)
upper_bound = np.percentile(simulated_prices, 50, axis=0)
lower_bound = np.percentile(simulated_prices, 50, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(mean_trajectory, color='blue', label='Mean Trajectory')
plt.fill_between(range(num_days), lower_bound, upper_bound, color='blue', alpha=0.1, label='90% Confidence Interval')  # Réduction de la transparence
plt.yscale('log')
plt.yticks([10**4, 10**5, 10**6], ['10,000', '100,000', '1,000,000'])
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Projection Cone of Bitcoin Prices')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='best')
plt.show()


max_drawdowns = []

for i in range(num_simulations):
    peak = simulated_prices[i, 0]
    drawdown = []
    for price in simulated_prices[i, :]:
        peak = max(peak, price)
        drawdown.append((peak - price) / peak)
    max_drawdowns.append(max(drawdown))

plt.figure(figsize=(10, 6))
plt.hist(max_drawdowns, bins=50, color='blue', alpha=0.7)
plt.xlabel('Max Drawdown (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Max Drawdowns')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


sorted_prices = np.sort(final_prices)
cumulative_probs = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)

plt.figure(figsize=(10, 6))
plt.plot(sorted_prices, cumulative_probs, marker='.', linestyle='-', color='blue')
plt.xlabel('Final Price ($)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Probability of Final Bitcoin Prices')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


historical_prices = bitcoin_data_clean['Price'].values
historical_dates = bitcoin_data_clean['Date'].values

plt.figure(figsize=(12, 6))
plt.plot(historical_dates, historical_prices, color='black', label='Historical Prices')
for i in range(100):  # On peut tracer un sous-ensemble des simulations pour une meilleure lisibilité
    plt.plot(pd.date_range(start=halving_dates[4], periods=num_days), simulated_prices[i, :], color='blue', alpha=0.1)
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Monte Carlo Simulations with Historical Bitcoin Prices')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()


# Par exemple, introduire un choc à mi-parcours
for i in range(num_simulations):
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
plt.show()


price_variances = np.var(simulated_prices, axis=0)

plt.figure(figsize=(12, 6))
plt.imshow(price_variances.reshape(1, -1), aspect='auto', cmap='viridis')
plt.colorbar(label='Variance')
plt.title('Heatmap of Price Variances Over Time')
plt.xlabel('Days')
plt.ylabel('Variance')
plt.show()


# Calculate VaR at 5%
var_5_percent = np.percentile(final_prices, 5)
# Calculate Expected Shortfall (CVaR)
cvar_5_percent = np.mean(final_prices[final_prices <= var_5_percent])

print(f"5% VaR: ${var_5_percent:,.0f}")
print(f"5% CVaR: ${cvar_5_percent:,.0f}")

plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, color='blue', alpha=0.7, label='Final Prices')
plt.axvline(var_5_percent, color='red', linestyle='--', label='5% VaR')
plt.axvline(cvar_5_percent, color='orange', linestyle='--', label='5% CVaR')
plt.xlabel('Final Price ($)')
plt.ylabel('Frequency')
plt.title('5% VaR and CVaR of Final Bitcoin Prices')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(historical_dates, historical_prices, color='black', label='Historical Prices')
plt.plot(pd.date_range(start=halving_dates[4], periods=num_days), mean_trajectory, color='blue', label='Mean Simulation Trajectory')
plt.fill_between(pd.date_range(start=halving_dates[4], periods=num_days), lower_bound, upper_bound, color='blue', alpha=0.3, label='90% Confidence Interval')
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Backtesting with Historical Data')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


import statsmodels.api as sm

quantiles = [0.1, 0.5, 0.9]
regression_results = []

for quantile in quantiles:
    quantile_prices = np.percentile(simulated_prices, quantile * 100, axis=0)
    model = sm.OLS(quantile_prices, sm.add_constant(range(num_days)))
    result = model.fit()
    regression_results.append(result)

plt.figure(figsize=(12, 6))
plt.plot(mean_trajectory, color='blue', label='Mean Trajectory')

for i, quantile in enumerate(quantiles):
    plt.plot(regression_results[i].fittedvalues, linestyle='--', label=f'{quantile * 100}% Quantile Regression')

plt.yscale('log')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Quantile Regression of Simulated Bitcoin Prices')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


# Appliquer différents chocs après 180 jours
shock_scenarios = [-0.25, -0.5, -0.75]  # Chutes de 20%, 50%, 80%
shock_day = 180

plt.figure(figsize=(12, 6))

for shock in shock_scenarios:
    shocked_prices = np.copy(simulated_prices)
    shocked_prices[:, shock_day:] *= (1 + shock)
    mean_shocked_trajectory = np.mean(shocked_prices, axis=0)
    plt.plot(mean_shocked_trajectory, label=f'Shock {int(shock * 100)}% on Day {shock_day}')

plt.plot(mean_trajectory, color='blue', label='No Shock')
plt.yscale('log')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Stress Testing with Different Scenarios')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()



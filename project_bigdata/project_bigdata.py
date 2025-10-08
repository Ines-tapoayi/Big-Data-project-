# PROJET BIG DATA - Inès TAPOAYI et Bô GUEGANNO (M2 IEF)

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import seaborn as sns

from tabulate import tabulate
from statsmodels.tsa.stattools import ccf
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ÉTAPE 1 : FORMATAGE DES DONNÉES

# Import des jeux de données 
# CSV généré par le code api_reddit.py
reddit = pd.read_csv(
    "reddit_posts.csv",
    sep=";",
    engine="python",
    on_bad_lines="skip",
    encoding="utf-8")

# CSV généré par le code api_coinbase.py
bitcoin = pd.read_csv(
    "bitcoin_prices.csv",
    sep=",",
    engine="python",
    on_bad_lines="skip",
    encoding="utf-8")

# Recherche et suppression des doublons
duplicates_id = reddit.duplicated(subset="id", keep=False)
reddit_clean = reddit.drop_duplicates(subset="id", keep="first")
duplicates_id_after = reddit_clean.duplicated(subset="id", keep=False)

# Conversion des dates
reddit["date"] = pd.to_datetime(reddit["created_utc"], unit='s', errors='coerce').dt.floor("D")
bitcoin["date"] = pd.to_datetime(bitcoin["time"], format="%Y-%m-%d", errors="coerce")

# Agrégation journalière des publications Reddit
daily_reddit = reddit.groupby("date").agg(
    n_posts=('id', 'count'),
    score_mean=('score', 'mean'),
    score_sum=('score', 'sum'),
    num_comments_mean=('num_comments', 'mean'),
    num_comments_sum=('num_comments', 'sum'))

# Création du dataset
dataset = pd.merge(daily_reddit, bitcoin, left_index=True, right_on="date", how="inner")
dataset = dataset.set_index("date")

# Vérification des valeurs manquantes
dataset.isnull().sum()
dataset = dataset.dropna(subset=['log_return'])
print(dataset)


# ÉTAPE 2 : STANDARDISATION DES VARIABLES

cols_to_standardize = ['n_posts',
                       'score_mean',
                       'score_sum',
                       'num_comments_mean',
                       'num_comments_sum',
                       'log_return']

scaler = StandardScaler()
dataset_standardized = dataset.copy()
dataset_standardized[cols_to_standardize] = scaler.fit_transform(dataset[cols_to_standardize])


# ÉTAPE 3 : ANALYSE DE L'ENGAGEMENT

cols = ['log_return',
        'n_posts',
        'score_mean',
        'score_sum',
        'num_comments_mean',
        'num_comments_sum']

reddit_cols = ['n_posts',
               'score_mean',
               'score_sum',
               'num_comments_mean',
               'num_comments_sum']

# Calcul des corrélations globales
corr = dataset_standardized[cols].corr()
corr_log_return = corr.loc[['log_return']].T.round(3)

print(tabulate(corr_log_return, headers=["Variables", "Corrélation"], tablefmt="psql"))

# Visualisation graphique
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})
plt.xticks(rotation=45, ha="right")
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()

# Calcul des corrélations glissantes

# Comparaison des fenêtres
windows = [3, 5, 7]

plt.figure(figsize=(12, 6))

for w in windows:
    rolling_corr = dataset_standardized['log_return'].rolling(w).corr(dataset_standardized['n_posts'])
    plt.plot(rolling_corr, label=f"Fenêtre de {w} jours")

plt.axhline(0, color="grey", linestyle="--", linewidth=1)
plt.title("Comparaison des corrélations glissantes (log_return vs n_posts)")
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

plt.show()

# Choix d'une fenêtre de 5 jours
threshold = 0.5
window = 5

rolling_corr = pd.DataFrame(index=dataset_standardized.index)
for col in reddit_cols:
    rolling_corr[col] = dataset_standardized['log_return'].rolling(window=window).corr(dataset_standardized[col])

# Visualisation graphique
n_cols = 2
n_rows = int(np.ceil(len(reddit_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2*n_rows), sharey=True) 
axes = axes.flatten()

for i, col in enumerate(reddit_cols):
    axes[i].plot(rolling_corr.index, rolling_corr[col], label=f'{col}', color='tab:blue')
    axes[i].axhline(0, color='grey', linestyle='--')
    
    strong_corr = rolling_corr[col].abs() > threshold
    axes[i].fill_between(rolling_corr.index, -1, 1, where=strong_corr, color='grey', alpha=0.2)
    axes[i].legend(loc='upper right')
    
    axes[i].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    axes[i].tick_params(axis='x')

for j in range(len(reddit_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Corrélation croisée
lags = range(-5, 6)

plt.figure(figsize=(12, 6))

for i, var in enumerate(reddit_cols, 1):
    ccf_values = sm.tsa.stattools.ccf(dataset_standardized[var], dataset_standardized["log_return"])
    
    plt.subplot(3, 2, i)
    plt.bar(lags, ccf_values[:len(lags)], color="lightsteelblue")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"{var} vs log_return")
    plt.xlabel("Décalage (jours)")
    plt.ylabel("Corrélation")

plt.tight_layout()
plt.show()


# ÉTAPE 4 : ANALYSE DE SENTIMENT

analyzer = SentimentIntensityAnalyzer()

# Nettoyage du texte
def clean_text(text):

    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def comments_sentiment(raw_text):

    if pd.isna(raw_text) or str(raw_text).strip() == "":
        return np.nan
    comments = [clean_text(c) for c in str(raw_text).split("|||") if clean_text(c)]
    if len(comments) == 0:
        return np.nan
    scores = [analyzer.polarity_scores(c)["compound"] for c in comments]
    return float(np.mean(scores))

# Calcul des scores et des pondérations
reddit["title_clean"] = reddit["title"].apply(clean_text)
reddit["sent_title"] = reddit["title_clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
reddit["sent_comments"] = reddit["top_comments"].apply(comments_sentiment)
reddit["sent_mean"] = reddit[["sent_title", "sent_comments"]].mean(axis=1)
reddit["sent_weighted_score"] = reddit["sent_mean"] * reddit["score"]
reddit["sent_weighted_comments"] = reddit["sent_mean"] * reddit["num_comments"]

# Agrégation journalière
daily_reddit_full = reddit.groupby("date").agg(
    n_posts=('id', 'count'),
    score_mean=('score', 'mean'),
    score_sum=('score', 'sum'),
    num_comments_mean=('num_comments', 'mean'),
    num_comments_sum=('num_comments', 'sum'),
    sent_comments_mean=('sent_comments', 'mean'),
    sent_comments_median=('sent_comments', 'median'),
    sent_title_mean=('sent_title', 'mean'),
    sent_mean=('sent_mean', 'mean'),
    sent_weighted_sum=('sent_weighted_score', 'sum'))

# Moyenne pondérée du sentiment par le score Reddit
daily_reddit_full["sent_weighted_mean"] = (
    daily_reddit_full["sent_weighted_sum"] / daily_reddit_full["score_sum"])

# Moyenne pondérée du sentiment par le nombre de commentaires
sent_weighted_comments = (
    reddit.groupby("date")["sent_weighted_comments"].sum() /
    reddit.groupby("date")["num_comments"].sum())

daily_reddit_full["sent_weighted_comments"] = sent_weighted_comments

# Création d'un nouveau dataset
dataset_full = pd.merge(daily_reddit_full, bitcoin, left_index=True, right_on="date", how="inner")
dataset_full = dataset_full.set_index("date")

# Standardisation des variables
cols_to_standardize_full = ["n_posts",
                            "score_mean",
                            "score_sum",
                            "num_comments_mean",
                            "num_comments_sum",
                            "sent_comments_mean",
                            "sent_comments_median",
                            "sent_title_mean",
                            "sent_mean",
                            "sent_weighted_mean",
                            "sent_weighted_comments",
                            "log_return"]

scaler = StandardScaler()
dataset_full_standardized = dataset_full.copy()
dataset_full_standardized[cols_to_standardize_full] = scaler.fit_transform(dataset_full[cols_to_standardize_full])

# Calcul des corrélations entre les scores de sentiment et log_return
sent_cols = ["sent_comments_mean","sent_comments_median","sent_title_mean","sent_mean","sent_weighted_mean","sent_weighted_comments"]
corr_sentiment = dataset_full_standardized[sent_cols + ["log_return"]].corr()

corr_df = corr_sentiment[["log_return"]].reset_index()
corr_df.columns = ["Variables", "Corrélation"]
corr_df = pd.concat([corr_df[corr_df["Variables"]=="log_return"], corr_df[corr_df["Variables"]!="log_return"]], ignore_index=True)

print(tabulate(corr_df, headers=["Variables", "Corrélation"], tablefmt="psql", floatfmt=".3f", showindex=False))

# Calcul des corrélations glissantes
rolling_corr = pd.DataFrame(index=dataset_full_standardized.index)

for col in sent_cols:
    rolling_corr[col] = (
        dataset_full_standardized["log_return"]
        .rolling(window=5)
        .corr(dataset_full_standardized[col]))

# Visualisation graphique
fig, axes = plt.subplots(3, 2, figsize=(14, 6), sharex=True)
axes = axes.flatten()

for i, col in enumerate(sent_cols):
    ax1 = axes[i]
    ax2 = ax1.twinx()

    # Courbe corrélation
    l1, = ax1.plot(
        rolling_corr.index, rolling_corr[col],
        label=f"{col}",
        color="orange", linewidth=2)
    
    ax1.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Courbe log_return
    l2, = ax2.plot(
        dataset_full_standardized.index,
        dataset_full_standardized["log_return"],
        label="log_return",
        color="blue", alpha=0.5, linewidth=1.5)

    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8, frameon=False)

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

for j in range(len(sent_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Corrélation croisée 
lags = range(-5, 6)

plt.figure(figsize=(12, 6))

for i, var in enumerate(sent_cols, 1):
    data = dataset_full_standardized[[var, "log_return"]].dropna()
    ccf_values = sm.tsa.stattools.ccf(data[var], data["log_return"])[:len(lags)]
    
    plt.subplot(3, 2, i)
    plt.bar(lags, ccf_values, color="lightsteelblue")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"{var} vs log_return")
    plt.xlabel("Décalage (jours)")
    plt.ylabel("Corrélation")

plt.tight_layout()
plt.show()
import requests
from datetime import datetime
import pandas as pd
import numpy as np

# Configuration du fichier
url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
granularity = 86400  # 1 jour
start_date = datetime(2025, 9, 4)
end_date = datetime(2025, 10, 5)

# Récupération des données
params = {
    "start": start_date.isoformat() + "Z",
    "end": end_date.isoformat() + "Z",
    "granularity": granularity
}
response = requests.get(url, params=params)
data = response.json()

# Transformation des données en DataFrame
df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
df["time"] = pd.to_datetime(df["time"], unit="s") 
df = df.sort_values("time")

# Calcul du rendement logarithmique à partir du prix moyen journalier
df['log_return'] = np.log((df['high'] + df['low'] + df['close'])/3 / ((df['high'] + df['low'] + df['close'])/3).shift(1))

# Sauvegarde en format CSV
df.to_csv("bitcoin_prices.csv", index=False)

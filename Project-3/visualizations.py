import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import add_features

os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/smart_city_traffic_sample.csv")
df = add_features(df)

plt.figure(figsize=(8, 5))
plt.hist(df["traffic_demand"], bins=30)
plt.title("Traffic Demand Distribution")
plt.xlabel("Traffic Demand")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/traffic_demand_distribution.png")
plt.close()

hourly = df.groupby("hour")["traffic_demand"].mean()
plt.figure(figsize=(8, 5))
hourly.plot(marker="o")
plt.title("Average Hourly Traffic Demand")
plt.xlabel("Hour")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.savefig("reports/hourly_traffic_analysis.png")
plt.close()

numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()

print("Charts saved inside reports/ folder.")

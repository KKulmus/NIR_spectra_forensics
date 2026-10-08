import pandas as pd
from paths import PROJECT_ROOT
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
df = pd.read_excel(PROJECT_ROOT / "dataset/metadata.xlsx")
out = df.groupby('type').size()
out = out.sort_values(ascending = True)
print(out)
print(df[df['set']=='K'].groupby('type').size())
#cathinones = df[df['type'] == 'Cathion']
#print(cathinones['code'].unique())          # eindeutige Proben
#print(cathinones['code'].nunique())
print(out.value_counts())  # Verteilung pro Cathinon
#print(df[df["code"] == "C9"]["component"].iloc[0])
#print(df.head())

plt.barh(out.index, out.values, zorder = 3, align='center', color='skyblue')
plt.suptitle("Sample Distribution by Substance Class", fontsize = 13)
plt.title("Cocaine dominates with 208 out of 526 entries (39.6%) - majority baseline", fontsize = 9)
plt.xlabel("Number of Samples")
plt.ylabel("Substance Class")
plt.grid(True, axis='x', zorder = 0)
plt.savefig(PROJECT_ROOT / "figs" / "sample_distribution.png", dpi=300, bbox_inches="tight")
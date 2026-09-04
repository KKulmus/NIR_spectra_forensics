import pandas as pd
pd.set_option('display.max_rows', None)
df = pd.read_excel("../dataset/metadata.xlsx")
out = df.groupby('type').size()
out = out.sort_values(ascending = False)
print(out)
#cathinones = df[df['type'] == 'Cathion']
#print(cathinones['code'].unique())          # eindeutige Proben
#print(cathinones['code'].nunique())
#print(cathinones['component'].value_counts())  # Verteilung pro Cathinon
#print(df[df["code"] == "C9"]["component"].iloc[0])
#print(df.head())

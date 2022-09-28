from fredapi import Fred
import pandas as pd
import os


fred_key = os.environ['fred_key']
fred = Fred(api_key=fred_key)

data = fred.get_series('SDXRSA')
df = pd.DataFrame(data=data, columns=['y'])
df['unique_id'] = 0
df = df.reset_index()
df = df.rename(columns={"index": "ds"})

df.to_csv('data.csv', index=False)
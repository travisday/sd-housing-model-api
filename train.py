import pandas as pd
from autots import AutoTS
from os.path import exists

model = AutoTS(forecast_length=12,
               frequency='infer',
               model_list="superfast",
               ensemble='simple')

if exists('autots_model.csv'):
  model.import_template('autots_model.csv', method="addon")

if exists('data.csv'):
  df = pd.read_csv('data.csv')
  df['ds'] = pd.to_datetime(df['ds'])
  model = model.fit(df, date_col='ds', value_col='y', id_col=None)
  model.export_template(
    'autots_model.csv',
    models="best",
    n=1,
    max_per_model_class=6,
    include_results=True,
  )

import pandas as pd
from autots import AutoTS
from os.path import exists

model = AutoTS(forecast_length=12,
               frequency='infer',
               model_list="superfast",
               ensemble='simple')

if exists('data.csv'):
  df = pd.read_csv('data.csv')
  model = model.fit(df, date_col='ds', value_col='y')
  model.export_template('model.csv', models='best',
                      n=15, max_per_model_class=3)
  prediction = model.predict()
  preds = prediction.forecast

  preds = preds.reset_index()
  preds = preds.rename(columns={"index": "ds"})
  preds.to_csv('forecast.csv', index=False)
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from os.path import exists
from autots import AutoTS

app = FastAPI()

def train():
  model = AutoTS(forecast_length=12,
               frequency='infer',
               model_list="superfast",
               ensemble='simple')

  if exists('data.csv'):
    df = pd.read_csv('data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    model = model.fit(df, date_col='ds', value_col='y', id_col=None)
    model.export_template('model.csv', models='best',
                        n=15, max_per_model_class=3)
    prediction = model.predict()
    preds = prediction.forecast

    preds = preds.reset_index()
    preds = preds.rename(columns={"index": "ds"})
    preds.to_csv('forecast.csv', index=False)

@app.get("/")
async def root():
  return "SD Housing Price Predictor"


@app.get("/pred")
async def predictions():
  df = pd.read_csv('forecast.csv')
  df['ds'] = pd.to_datetime(df['ds'])
  ds = jsonable_encoder(df['ds'].values.tolist())
  y = jsonable_encoder(df['y'].values.tolist())
  out = {'ds': ds, 'y': y}
  return JSONResponse(content=jsonable_encoder(out))


@app.get("/data")
async def get_data():
  df = pd.read_csv('data.csv')
  df['ds'] = pd.to_datetime(df['ds'])
  ds = jsonable_encoder(df['ds'].values.tolist())
  y = jsonable_encoder(df['y'].values.tolist())
  out = {'ds': ds, 'y': y}
  return JSONResponse(content=jsonable_encoder(out))

@app.get("/train")
async def root():
  train()
  return "Trained!"


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8080)

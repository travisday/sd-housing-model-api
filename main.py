import pandas as pd
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from os.path import exists
from autots import AutoTS

app = FastAPI()

def autots_predict():
  model = AutoTS(forecast_length=12,
                 frequency='infer',
                 model_list="superfast",
                 ensemble='simple')

  if exists("autots_model.csv"):
    model.import_template("autots_model.csv", method="only")

  if exists('data.csv'):
    df = pd.read_csv('data.csv')
    df['ds'] = pd.to_datetime(df['ds'])

    model = model.fit(df, date_col='ds', value_col='y', id_col=None)
  
    prediction = model.predict(fail_on_forecast_nan=True)
    forecasts_df = prediction.forecast

    return jsonable_encoder(forecasts_df['y'].values.tolist())


@app.get("/")
async def root():
  return {"message": "Hello World"}


@app.get("/pred")
async def predictions():
  return JSONResponse(content=autots_predict())


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8080)

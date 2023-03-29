import pandas as pd
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

  if exists("model.csv"):
    model.import_template("model.csv", method="only")

    prediction = model.predict()
    forecast = prediction.forecast

    return jsonable_encoder(forecast['y'].values.tolist())


@app.get("/")
async def root():
  return {"message": "Hello World"}


@app.get("/pred")
async def predictions():
  return JSONResponse(content=autots_predict())


@app.get("/data")
async def get_data():
  df = pd.read_csv('data.csv')
  df['ds'] = pd.to_datetime(df['ds'])
  ds = jsonable_encoder(df['ds'].values.tolist())
  y = jsonable_encoder(df['y'].values.tolist())
  out = {'ds': ds, 'y': y}
  return JSONResponse(content=jsonable_encoder(out))


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8080)

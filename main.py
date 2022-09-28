from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
import uvicorn
from fastapi import FastAPI

app = FastAPI()

def train_model(df):
  autoARIMA = AutoARIMA()
  model = StatsForecast(df=df, 
                        models=[autoARIMA],
                        freq='M', n_jobs=-1)

def get_predictions():
  preds = model.forecast(12)
  return preds['AutoARIMA'].values


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/get_predictions")
async def root():
    return get_predictions()


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8000)

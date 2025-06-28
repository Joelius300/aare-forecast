import datetime

import pandas as pd
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from aare.utils import find_project_root
from lib import hello

# dict/None = don't know the type yet :)

def load_model():
    # load model (from pickle)
    # load feature set from yaml
    pass

def persist_metadata(run_ts: datetime.datetime):
    # store version (?), features, etc.
    pass

def fetch_external_data() -> dict:
    # pull from all registered external sources
    
    # ACTUALLY need to decided if we want to fetch _all_ data even if we're not using it for the model (not in features)
    # because that would build a dataset of data we can use later for services that don't offer historical forecasts
    # like the flow prediction of BAFU. -> yes do that :)
    pass

def get_inference_data(features: dict, external_data: dict):
    # get actual features from the registry
    # take what you can from influx, the rest must be in external_data
    pass

def persist_data(run_ts: datetime.datetime, data: dict):
    # store all the data together with the run_ts as identification
    pass

def predict(model: GlobalForecastingModel, data: dict) -> pd.DataFrame:
    # use the data to predict the coming temperature
    pass
    
def persist_prediction(run_ts: datetime.datetime, prediction: pd.DataFrame):
    # store prediction to db
    pass

if __name__ == "__main__":
    hello()
    print(f"Project Root: {find_project_root()}")

    run_ts = datetime.datetime.now(datetime.UTC)
    model, features = load_model()
    
    persist_metadata(run_ts)
    external_data = fetch_external_data()
    persist_data(run_ts, external_data)
    data = get_inference_data(features, external_data)
    
    prediction = predict(model, data)
    persist_prediction(run_ts, prediction)

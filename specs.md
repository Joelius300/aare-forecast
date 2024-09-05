## Motivation

At the moment, only the current values are visible in the aare.guru app/website.
Users of aare.guru should be able to view a rough forecast of the discharge and temperature in the coming days.

## Scope

For the initial implementation, the scope is restricted heavily.
Depending on the needs of the users (if it's actually deployed) and my motivation and available time, additional features
outside of this scope can be implemented later.

- Only forecast the location "2135, Bern, Schönau". Next on the list would be "2030, Thun".
- Only forecast the water temperature (there already is a forecast for the discharge by BAFU).
- Don't include external data sources in the beginning (e.g. existing forecasts for air temperature, rainfall and discharge).

## Forecast specifications

These are initial values, set from a users perspective.
They can be adjusted if experimentation suggests to or there are technical limitations.

- Resolution: 1h
- Horizon: 4 days = 96 hours (roughly the same as the existing discharge forecast)
- Probabilistic forecast if possible (BAFU uses 25 and 75 quantiles)

## Evaluation

Both MAE and RMSE are highly relevant and will be used for comparing different methods and models. There is no need for MASE, RMSSE or similar.

## Possible experiments (dump)

- Seasonality testing; should have a weak daily periodicity
- Baselines: Naive, MeanNaive, SNaive
- Evaluate TimesFM from Google as strong baseline
- Univariate foreasting only at first
- Models to try: ARIMA, GRU/LSTM, Linear Regression
  - More models: TSMixer, TFT (but we're trending away from transformers)
  - Could also fine-tune TimesFM
- Multivariate -- past covariates, include time encoding, air temperature and discharge
- Multivariate -- past covariates, include values from locations upstream (e.g. Thun)
- Multivariate -- future covariates, include weather forecasts (temperature and precipitation) from MeteoSwiss
- Multivariate -- future covariates, include discharge forecast from BAFU
- Different losses: mae, mse, (regularized methods)
- Out of scope:
  - Global model that can forecast different locations (location is a static covariate)
  - Try forecasting the discharge as well, see how close you get to the one from BAFU (they use 21 black-, white-, and gray-box models tuned by hydrologists)

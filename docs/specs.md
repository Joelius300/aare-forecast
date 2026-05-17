## Motivation

> [!NOTE]
> This was the initial spec document I wrote for myself when starting the project
> long before I got in touch with aare.guru.

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
- Forecasts are made once a day at midnight, so models don't need to be able to start forecasting from 3pm.
  - Once that's done, test hourly (evaluate with stride 23) and adjust training if it performs badly.
- Probabilistic forecast if possible (BAFU uses 25 and 75 quantiles)

## Evaluation

Both MAE and RMSE are highly relevant and will be used for comparing different methods and models. There is no need for MASE, RMSSE or similar.

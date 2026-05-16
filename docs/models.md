# Aare Oraku Models

Description of the models inside the Aare Oraku. Models are named with their use case and their version.
The model version is a numerical two-component version where the first component is bumped when the model
architecture changes (e.g. new type, new input or output variable, etc.) and the second is bumped when
the training data or the training parameters change, or when library/compatibility updates are made.

This notebook gives insights into the accuracy of these models: <https://joelius300.github.io/aare-forecast>
(takes a while to load). Note that in this first version of Aare Oraku, most things are centered around this
single use case with just one location, so a lot can and will change when other use cases and locations are introduced.

## Nowcasting temp[erature]

This model use case is for short-term river temperature forecasts (1-2 days) with 1h resolution.
It's the first and only model at the moment, but there are plans to make models for longer horizons
and potentially even river flow.

> [!IMPORTANT]
> "nowcasting" isn't a good name for this. I had naively thought that the water temperature moves so slowly that 1h
> would be the highest resolution we need but closer examination of live forecasts in early summer with volatile weather
> has shown that actual nowcasting e.g. 10min resolution for the next 3-6 hours might have its uses too.
> If I ever implement actual nowcasting, this model might be renamed to shortcasting or something similar.

### v1.0

Good enough for a first model, but definitely has lots of room for improvement.

- **Architecture:** Autoregressive LR model
- **Inputs features:** Air temperature (with non-linear transforms) at lags up to 24h
- **Locations:** Only Bern, Schönau (2135)
- **Accuracy:** 50% of forecasts have a mean absolute error across all the predicted 36h of **0.4 °C or less**.
  **80% of 0.72 °C or less**. For just the first hour, 75% are below 0.23°C.
  In the worst summer periods, errors are regularly between 0.5 and 1.5 °C at larger lags.
  _This is only on measurements, performance will degrade when using forecasts as inputs (like during inference)!_

### v1.1

> [!IMPORTANT]
> This version is released together with a change in the target variable. The switch from hourly mean to point in time
> forecasts was made for better alignment with the use case and visualization at aare.guru.

Same as v1.0, with the following differences:

- temp_bern diff_threshold (used for outlier detection) increased from 0.6°C to 1.0°C to handle volatile spring/early
  summer weather.
- switch from hourly mean values to subsampling to avoid a target and covariate mismatch when a) plotting forecasts
  alongside point in time measurement values and b) using meteotest forecasts as inputs, which are also aligned
  to points in time without mean aggregation.

These two changes improved the accuracy of the model as well.

- 50% of forecasts have a mean absolute error across all the predicted 36h of **0.38 °C or less**.
  **80% of 0.69 °C or less**.
- For just the first hour, 75% are below 0.13°C (this is the biggest win of this update).

### v1.2

Same as v1.1, with the following differences:

- update to darts v0.44.1

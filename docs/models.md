# Aare Oraku Models

Description of the models inside the Aare Oraku. Models are named with their use case and their version.
The model version is a numerical two-component version where the first component is bumped when the model
architecture changes (e.g. new type, new input or output variable, etc.) and the second is bumped when
the training data or the training parameters change, or when library/compatibility updates are made.

This notebook gives insights into the accuracy of these models: <https://joelius300.github.io/aare-forecast>
(takes a while to load). Note that in this first version of Aare Oraku, most things are centered around this
single use case with just one location, so a lot can and will change when other use cases and locations are introduced.

## Nowcasting temp[erature]

This model use case is for short-term river temperature forecasts (1-2 days).
It's the first and only model at the moment, but there are plans to make models for longer horizons
and potentially even river flow.

### v1.0

Good enough for a first model, but definitely has lots of room for improvement.

- **Architecture:** Autoregressive LR model
- **Inputs features:** Air temperature (with non-linear transforms) at lags up to 24h
- **Locations:** Only Bern, Schönau (2135)
- **Accuracy:** 50% of forecasts have a mean absolute error across all the predicted 36h of **0.4 °C or less**.
  **80% of 0.7 °C or less**. In the worst summer periods, errors are regularly between 0.5 and 1.5 °C at larger lags.
  _This is only on measurements, performance will degrade when using forecasts as inputs (like during inference)!_

class HydroFlow:
    # TODO
    # https://www.hydrodaten.admin.ch/plots/q_forecast/2135_q_forecast_de.json
    # in data[0] and data[1] is min/max (need to figure out which is which)
    # in data[2] are 25% and 75% quantiles ASC, then DESC, so twice as many data points
    # in data[3] is the median, so should have the same n as data[0] and data[1]
    # in data[4] are the true measurement values up to the point the forecast was made
    # it seems the prediction was made at the first data point or shortly before that, found no exact time
    # maybe send an email to BAFU to see where best to consume this data, given that we're pulling a plot and parsing
    # the data out of that, but I don't think they have an API for consumers for stuff like this.
    pass

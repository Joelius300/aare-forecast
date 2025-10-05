# Translation of measurements to external sources registered in the SourceRegistry.
# If a field of a specified measurement (e.g. the 'tt' field of the 'smn' measurement)
# is required in the future [for inference], then the forecast service will look in the
# here specified external source for it.
# Maybe later it needs to be split because smn/ABC can be taken from meteotest,
# but smn/XYZ needs to be taken from gugusglüngi. For now just measurements should be fine.

# Is there a smart way to reduce the number of magic strings here?
# Later I might want this to be in config files and then it's like this anyway.
# Also circular imports could get annoying for things like that.
MEAS_TRANS = {
    "smn": "meteotest",
}

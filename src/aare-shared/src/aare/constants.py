# location codes (hydro)
LOC_BERN = 2135  # Bern, Schönau
LOC_THUN = 2030
LOC_INT = 2457  # Interlaken; Ringgenberg, Goldswil
LOC_BRNZ = 2019  # Brienz; Brienzwiler
LOC_HAG = 2085  # Hagneck
LOC_BIEL = 2029  # Brügg, Aegerten
LOC_BRGG = 2016  # Brugg
# Olten does not exist in hydro network unfortunately, custom temp sensor for app

# location codes (meteo/smn)
LOC_BERN_SMN = "BER"
LOC_THUN_SMN = "THU"
LOC_INT_SMN = "INT"
LOC_BRNZ_SMN = "BRZ"
# TODO figure out which station aare.guru uses for those (Cressier? Grenchen?)
LOC_HAG_SMN = None  # Cressier?
LOC_BIEL_SMN = None  # Grenchen?
LOC_BRGG_SMN = None  # Buchs (AG), Lägern, Beznau?

# column names for raw influx hydro data
TIME = "_time"
TEMP = "temperature"

ANYTIME = "0"
"""To be used as period start when querying influx. Starting at 0 just returns all the data."""

RANDOM_SEED = 42
"""Fixed random seed to make experiments reproducible"""

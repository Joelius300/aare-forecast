"""
Constants for various ids, columns, etc.
Most of it is from the project's infancy and abstracted away, but it can still be used directly if needed.
"""

# location codes (hydro)
LOC_BERN = 2135  # Bern, Schönau
LOC_THUN = 2030
LOC_INT = 2457  # Interlaken; Ringgenberg, Goldswil
LOC_BRNZ = 2019  # Brienz; Brienzwiler
LOC_HAGN = 2085  # Hagneck
LOC_BIEL = 2029  # Brügg, Aegerten
LOC_BRGG = 2016  # Brugg
# Olten does not exist in hydro network unfortunately, custom temp sensor for app

# location codes (meteo/smn) / sometimes difficult to choose, these are the same as aare.guru
LOC_BERN_SMN = "BER"
LOC_THUN_SMN = "THU"
LOC_INT_SMN = "INT"
LOC_BRNZ_SMN = "MER"  # Meiringen instead of Brienz because that is close and has much more data
LOC_HAGN_SMN = "CRM"  # Cressier
LOC_BIEL_SMN = "GRE"  # Grenchen
LOC_BRGG_SMN = "BEZ"  # Beznau

# column names for raw influx hydro data (legacy)
TIME = "_time"
TEMP = "temperature"

ANYTIME = "0"
"""To be used as period start when querying influx. Starting at 0 just returns all the data."""

RANDOM_SEED = 42
"""Fixed random seed to make experiments reproducible"""

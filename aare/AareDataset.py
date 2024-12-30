from aare.constants import LOC_BERN
from aare.remote_existenz_store import RemoteExistenzStore


class AareDataset:
    def __init__(self, store: RemoteExistenzStore, val_split: str, test_split: str):
        self._store = store
        self._val_split = val_split
        self._test_split = test_split
        self._loc = LOC_BERN

    def _fetch_data(self, period: str | tuple[str, str]):
        return self._store.query_hydro(period, self._loc)

    def get_train(self):
        return self._fetch_data(("0", self._val_split))

    def get_val(self):
        return self._fetch_data((self._val_split, self._test_split))

    def get_test(self):
        return self._fetch_data(self._test_split)

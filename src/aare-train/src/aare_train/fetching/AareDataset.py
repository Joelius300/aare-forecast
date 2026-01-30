from aare.constants import LOC_BERN
from aare_train.params import read_params
from aare_train.fetching.remote_existenz_store import RemoteExistenzStore


class AareDataset:
    """
    Direct access to the raw train, val and test data for the target of this project (water temperature in berne).
    Very inflexible, prefer RemoteExistenzStore, Feature and FeatureSet.
    """

    def __init__(
        self,
        store: RemoteExistenzStore,
        train_split: str,
        val_split: str,
        test_split: str,
    ):
        self.store = store
        self._train_split = train_split
        self._val_split = val_split
        self._test_split = test_split
        self._loc = LOC_BERN

    @classmethod
    def from_conf(cls):
        store = RemoteExistenzStore()
        params = read_params()
        return AareDataset(store, **params["split"])

    def _fetch_data(self, period: str | tuple[str, str]):
        return self.store.query_hydro(period, self._loc)

    def get_train(self):
        return self._fetch_data((self._train_split, self._val_split))

    def get_val(self):
        return self._fetch_data((self._val_split, self._test_split))

    def get_test(self):
        return self._fetch_data(self._test_split)  # no upper bound ( = now() )

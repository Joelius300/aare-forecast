from collections.abc import Callable

from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.models import GRUTrainer, LRTrainer, TSMixerTrainer, BaseTrainer
from aare_train.params import Params

MODEL_TRAINERS: dict[str, Callable[[Params, FeatureIdentifiers], BaseTrainer]] = {
    "GRU": GRUTrainer,
    "LR": LRTrainer,
    "TSMIXER": TSMixerTrainer,
}
MODEL_TYPES = tuple(k.lower() for k in MODEL_TRAINERS.keys())

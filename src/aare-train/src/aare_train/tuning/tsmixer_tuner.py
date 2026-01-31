from typing import override

from aare_train.evaluation.eval_metric import EvalMetric
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params.params_types import Params
from aare_train.tuning.base_tuner import BaseTuner, ModelType
from aare_train.models.tsmixer_trainer import TSMixerTrainer
from optuna import Trial


class TSMixerTuner(BaseTuner):
    def __init__(
        self, model_name: str, params: Params, features: FeatureIdentifiers, batch_size: int, max_n_epochs: int
    ):
        super().__init__(model_name, params, features)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_n_epochs = max_n_epochs

    @override
    def get_model(self, trial: Trial) -> ModelType:
        # suggest hyperparameters
        input_chunk_length = trial.suggest_int("input_chunk_length", 1, 48, step=6)
        output_chunk_length = trial.suggest_int("output_chunk_length", 1, 48, step=6)
        hidden_size = trial.suggest_int("hidden_size", 1, 256, log=True)
        ff_size = trial.suggest_int("ff_size", 1, 256, log=True)
        num_blocks = trial.suggest_int("num_blocks", 1, 10)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
        add_day_enc = trial.suggest_categorical("add_day_enc", [True, False])
        add_year_enc = trial.suggest_categorical("add_year_enc", [True, False])

        # create trainer with suggested hyperparameters and pruning-enabled trainer kwargs
        trainer = TSMixerTrainer(
            self.params,
            self.features,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            ff_size=ff_size,
            num_blocks=num_blocks,
            dropout=dropout,
            lr=lr,
            batch_size=self.batch_size,
            max_n_epochs=self.max_n_epochs,
            add_day_enc=add_day_enc,
            add_year_enc=add_year_enc,
            model_name=self.model_name,
            pl_trainer_kwargs=self.get_trainer_params_with_pruning(trial, self.model_name),
        )

        # use the trainer's data and build model
        self._copy_data_from_trainer(trainer)
        return trainer.build_model()

    @override
    def get_optim_vars(self, metrics: EvalMetric, model: ModelType):
        # this is what will be minimized by optuna
        return metrics.mae

    @override
    def initial_trials(self):
        small = {
            "input_chunk_length": 24,
            "output_chunk_length": 24,
            "hidden_size": 16,
            "ff_size": 16,
            "num_blocks": 1,
            "dropout": 0,
            "lr": 1e-4,
            "add_day_enc": False,
            "add_year_enc": False,
        }
        medium = {
            "input_chunk_length": 24,
            "output_chunk_length": 24,
            "hidden_size": 64,
            "ff_size": 64,
            "num_blocks": 2,
            "dropout": 0.1,
            "lr": 1e-4,
            "add_day_enc": False,
            "add_year_enc": False,
        }

        yield small
        yield medium

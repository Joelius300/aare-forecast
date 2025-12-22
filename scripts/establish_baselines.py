import logging

from darts.models import GlobalNaiveSeasonal, GlobalNaiveAggregate

from aare.evaluation.pipeline import evaluation_pipeline_uni
from aare.params import read_params

logger = logging.getLogger(__name__)


def main():
    params = read_params()
    horizon = params["general"]["forecast_horizon"]
    models = {
        "LOCF": GlobalNaiveSeasonal(input_chunk_length=1, output_chunk_length=1),
        # daily seasonality
        "SNAIVE": GlobalNaiveSeasonal(input_chunk_length=24, output_chunk_length=1),
        # weekly mean
        "MEAN": GlobalNaiveAggregate(input_chunk_length=7 * 24, output_chunk_length=horizon),
    }

    evaluation_pipeline_uni(models, horizon, params["validation"], params["general"]["timezone"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

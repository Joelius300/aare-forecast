import logging
import sys

from darts.models import GlobalNaiveSeasonal, GlobalNaiveAggregate

from aare_train.evaluation.pipeline import evaluation_pipeline_uni
from aare_train.params import read_params

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

    use_test = len(sys.argv) >= 2 and sys.argv[1] == "--test"
    evaluation_pipeline_uni(models, params, use_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

from typing import TypedDict

from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from aare.feature_identifiers import FeatureIdentifiers


# either store this as pickle -> cannot read it without python
# or store as json, but then don't directly use the type but instead a string id (will not support many anyway)
# or serialize type to json with options below
class AareModel(TypedDict):
    model_path: str
    model_cls: type[GlobalForecastingModel]  # can serialize, just look below for two options
    features: FeatureIdentifiers
    # TODO not sure about these two, but it would be nice to have this directly readable/visible next to the model
    name: str
    version: str


# ----

import json  # noqa: E402
import importlib  # noqa: E402

# Example dictionary
my_dict = {
    "name": "example",
    "description": "a class example",
    "cls": "collections.OrderedDict",  # or __module__ + "." + __qualname__
}

# Save to JSON
with open("data.json", "w") as f:
    json.dump(my_dict, f, indent=4)


# Later, to load and restore the class
def get_class_by_name(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# Load from JSON
with open("data.json") as f:
    loaded = json.load(f)
    cls = get_class_by_name(loaded["cls"])

# ----


class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {"__class__": True, "path": o.__module__ + "." + o.__qualname__}
        return super().default(o)


def class_decoder(obj):
    if "__class__" in obj:
        return get_class_by_name(obj["path"])
    return obj


# Usage
json_str = json.dumps(my_dict, cls=ClassEncoder, indent=4)
my_dict_back = json.loads(json_str, object_hook=class_decoder)

# ----

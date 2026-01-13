"""This module is to read
configuration parameters values
"""

import yaml

PARAMS_FILE = "params.yaml"
METHODS_PARAMS_FILE = "methods_params.yaml"

with open(PARAMS_FILE) as f:
    params = yaml.safe_load(f)["params"]

with open(METHODS_PARAMS_FILE) as f:
    methods_params = yaml.safe_load(f)

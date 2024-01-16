"""Config utilities."""
from typing import Dict
import yaml

def load_config(config_path: str = "configs/application.yaml") -> Dict:
    """
    Load the config file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

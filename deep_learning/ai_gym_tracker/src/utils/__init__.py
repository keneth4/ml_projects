"""Initialize configuration."""
from .config import load_config

config = load_config("configs/application.yaml")

# Load counter config
counter_config = config['interface']['counter']

# Load message config
message_config = config['interface']['message']

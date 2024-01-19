"""Initialize configuration."""
from .config import load_config

config = load_config("configs/application.yaml")

# Load counter config
counter_config = config['interface']['counter']

# Load double counter config
double_counter_config = config['interface']['double_counter']

# Load message config
message_config = config['interface']['message']

# Load title config
title_config = config['interface']['title']

# Load sets config
sets_config = config['interface']['sets']

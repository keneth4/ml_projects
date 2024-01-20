"""Initialize configuration."""
from .config import load_config

config = load_config("configs/application.yaml")

# Load text configs
text_config = config['interface']['text_configs']
counter_config = text_config['counter']
double_counter_config = text_config['double_counter']
message_config = text_config['message']
title_config = text_config['title']
sets_config = text_config['sets']
timer_config = text_config['timer']
numeric_menu_config = text_config['numeric_menu']

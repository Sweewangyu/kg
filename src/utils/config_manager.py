import os
import yaml

class ConfigManager:
    _config = None

    @classmethod
    def get_config(cls):
        if cls._config is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            with open(config_path, "r", encoding="utf-8") as file:
                cls._config = yaml.safe_load(file)
        return cls._config

class ConfigManager:
    _config = {
        "agent": {
            "language": "auto",
            "chunk_char_limit": 1024,
            "chunk_overlap_sentences": 2,
        }
    }

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._config = {
                "agent": {
                    "language": "auto",
                    "chunk_char_limit": 1024,
                    "chunk_overlap_sentences": 2,
                }
            }
        return cls._config

    @classmethod
    def set_config(cls, config: dict):
        cls._config = config

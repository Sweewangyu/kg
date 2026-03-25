import argparse

from pipeline import Pipeline
from models.llm_def import LLMFactory
from utils.config_manager import ConfigManager
from utils.process import load_extraction_config
from utils.logger import logger


def main():
    logger.info("Starting knowledge graph extraction and reflection pipeline.")
    parser = argparse.ArgumentParser(description="Run joint knowledge graph extraction and reflection.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_extraction_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    model_config = config["model"]
    logger.info(f"Initializing LLM model with {model_config.get('model_name_or_path')} at {model_config.get('base_url', 'https://api.openai.com/v1')}")
    model = LLMFactory.create_llm(
        model_name_or_path=model_config.get("model_name_or_path"),
        api_key=model_config.get("api_key", ""),
        base_url=model_config.get("base_url", "https://api.openai.com/v1"),
    )
    pipeline = Pipeline(model)

    extraction_config = config["extraction"]
    ConfigManager.get_config()["agent"]["language"] = extraction_config.get("language", "auto")

    logger.info("Executing pipeline for extraction...")
    pipeline.get_extract_result(
        text=extraction_config["text"],
        use_file=extraction_config["use_file"],
        file_path=extraction_config["file_path"],
        show_trajectory=extraction_config["show_trajectory"],
    )
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()

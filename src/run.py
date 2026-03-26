import argparse

from pipeline import Pipeline
from models.llm_def import LLMFactory
from utils.config_manager import ConfigManager
from utils.process import load_extraction_config
from utils.logger import logger


def main():
    logger.info("Starting knowledge graph pipeline.")
    parser = argparse.ArgumentParser(description="Run knowledge graph extraction or reverse generation.")
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

    task_config = config["task"]
    extraction_config = config["extraction"]
    generation_config = config["generation"]
    agent_config = config["agent"]
    ConfigManager.set_config(
        {
            "agent": {
                "language": extraction_config.get("language", "auto"),
                "chunk_char_limit": agent_config["chunk_char_limit"],
                "chunk_overlap_sentences": agent_config["chunk_overlap_sentences"],
            }
        }
    )
    runtime_agent_config = ConfigManager.get_config()["agent"]
    logger.info(
        "Using chunk settings: chunk_char_limit=%s, chunk_overlap_sentences=%s",
        runtime_agent_config["chunk_char_limit"],
        runtime_agent_config["chunk_overlap_sentences"],
    )

    if task_config["type"] == "extraction":
        logger.info("Executing pipeline for extraction...")
        pipeline.get_extract_result(
            text=extraction_config["text"],
            use_file=extraction_config["use_file"],
            file_path=extraction_config["file_path"],
            show_trajectory=extraction_config["show_trajectory"],
            use_reflection=task_config["use_reflection"],
        )
    elif task_config["type"] == "generation":
        logger.info("Executing pipeline for reverse generation...")
        pipeline.get_generation_result(
            triplets=generation_config["triplets_json"],
            show_trajectory=generation_config["show_trajectory"],
        )
    else:
        raise ValueError(f"Unsupported task: {task_config['type']}")
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()

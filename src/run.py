import argparse
import os
import yaml
from pipeline import Pipeline
from typing import Literal
import models
from models.llm_def import LLMFactory
from utils.process import load_extraction_config

def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Run the extraction framefork.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load configuration
    config = load_extraction_config(args.config)
    # Model config
    model_config = config['model']
    
    try:
        model = LLMFactory.create_llm(
            model_type=model_config.get('category', 'OpenAIModel').replace('Model', '').lower(),
            model_name_or_path=model_config.get('model_name_or_path'),
            api_key=model_config.get('api_key', ''),
            base_url=model_config.get('base_url', 'https://api.openai.com/v1')
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    pipeline = Pipeline(model)
    # Extraction config
    extraction_config = config['extraction']
    # constuct config
    if 'construct' in config:
        construct_config = config['construct']
        result, trajectory = pipeline.get_extract_result(task=extraction_config['task'], instruction=extraction_config['instruction'], text=extraction_config['text'], output_schema=extraction_config['output_schema'], constraint=extraction_config['constraint'], use_file=extraction_config['use_file'], file_path=extraction_config['file_path'], truth=extraction_config['truth'], mode=extraction_config['mode'], update_case=extraction_config['update_case'], show_trajectory=extraction_config['show_trajectory'],
                                                               construct=construct_config, iskg=True) # When 'construct' is provided, 'iskg' should be True to construct the knowledge graph.
        return
    else:
        print("please provide construct config in the yaml file.")

    result, trajectory = pipeline.get_extract_result(task=extraction_config['task'], instruction=extraction_config['instruction'], text=extraction_config['text'], output_schema=extraction_config['output_schema'], constraint=extraction_config['constraint'], use_file=extraction_config['use_file'], file_path=extraction_config['file_path'], truth=extraction_config['truth'], mode=extraction_config['mode'], update_case=extraction_config['update_case'], show_trajectory=extraction_config['show_trajectory'])
    return

if __name__ == "__main__":
    main()

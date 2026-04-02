import os
import sys
import json

# Add src directory to sys.path so we can import from the kg project
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline import Pipeline
from models.llm_def import LLMFactory
from utils.config_manager import ConfigManager
from utils.logger import logger

def main():
    # 1. Initialize the LLM model using user provided parameters
    base_url = "http://10.246.99.82:11025/v1"
    model_name = "qwen3.5-35b-a3b-w8a8-mtp"
    api_key = "sk-1234"
    
    logger.info(f"Initializing LLM model: {model_name} at {base_url}")
    model = LLMFactory.create_llm(
        model_name_or_path=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    
    # 2. Configure runtime parameters required by Pipeline
    ConfigManager.set_config(
        {
            "agent": {
                "language": "zh",
                "chunk_char_limit": 1000,
                "chunk_overlap_sentences": 1,
            }
        }
    )
    
    pipeline = Pipeline(model)
    
    # 3. Process input and output JSONL files
    input_file = "input.json" # Please replace with your input file path
    output_file = "output.json" # The generated output file path
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Count total lines for progress logging
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        
    logger.info(f"Starting to process {total_lines} records from {input_file} -> {output_file}")
    
    # 4. Stream process line by line
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for idx, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line {idx}: {e}")
                continue
                
            logger.info(f"Processing item {idx} / {total_lines}...")
            
            # Transform relation dicts to triplets with subject/relation/object keys
            triplets = []
            for rel in item.get("relation", []):
                triplets.append({
                    "subject": rel.get("head"),
                    "relation": rel.get("relation"),
                    "object": rel.get("tail")
                })
            
            # Call the pipeline for generation
            try:
                generated_text, trajectory = pipeline.get_generation_result(
                    triplets=triplets,
                    show_trajectory=False
                )
                
                # Update the text field with the generated result
                item["text"] = generated_text
                
                # Write back immediately to prevent data loss
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush() # Ensure it's written to disk immediately
                
                logger.info(f"Successfully processed and saved item {idx}")
                
            except Exception as e:
                logger.error(f"Error during generation for item {idx}: {e}")
                
    logger.info(f"Finished processing all items. Results saved to {output_file}")

if __name__ == "__main__":
    main()

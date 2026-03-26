import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from openai import OpenAI
from utils.process import load_extraction_config


def main():
    parser = argparse.ArgumentParser(description="Test model connectivity using the YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "KG.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请只回复一行：OK",
        help="Prompt sent to the configured model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens used for the test request.",
    )
    args = parser.parse_args()

    config = load_extraction_config(args.config)
    model_config = config["model"]
    base_url = model_config["base_url"] or "https://api.openai.com/v1"

    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=base_url,
    )

    print(f"Model: {model_config['model_name_or_path']}")
    print(f"Base URL: {base_url}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print("Sending request...")

    response = client.chat.completions.create(
        model=model_config["model_name_or_path"],
        messages=[
            {"role": "user", "content": args.prompt},
        ],
        stream=False,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    message = response.choices[0].message
    content = (message.content or "").strip()
    reasoning = getattr(message, "reasoning_content", None)

    print(f"Finish reason: {response.choices[0].finish_reason}")
    print("Content:")
    print(content or "<empty>")
    if reasoning:
        print("Reasoning content:")
        print(reasoning.strip())


if __name__ == "__main__":
    main()

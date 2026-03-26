import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.llm_def import BaseEngine
from pipeline import Pipeline
from utils.config_manager import ConfigManager
from utils.process import extract_json_dict, load_extraction_config


class FakeEngine(BaseEngine):
    def __init__(self):
        super().__init__("fake")

    def get_chat_response(self, prompt: str) -> str:
        if "Triplets To Express" in prompt:
            return "The Legend of the Golden Gun and The Sacketts were both published in 1979. Louis L'Amour wrote the screenplay for the film The Shadow Riders and is a renowned author in the Western genre."
        if "## Extraction Result" in prompt:
            return """
            {
              "schema_compliance": 92,
              "faithfulness": 89,
              "completeness": 86,
              "normalization": 85,
              "overall_score": 88,
              "major_problems": ["invalid relation"],
              "minor_problems": ["invalid attribute"]
            }
            """
        return """
        {
          "entities": [
            {"name": "Alice", "type": "person"},
            {"name": "Bob", "type": "person"}
          ],
          "attributes": {
            "Alice": ["profession: engineer"]
          },
          "triples": [
            {"Head": "Alice", "Relation": "spouse of", "Tail": "Bob"}
          ]
        }
        """


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        ConfigManager._config = None
        self.pipeline = Pipeline(FakeEngine())

    def tearDown(self):
        ConfigManager._config = None

    def test_pipeline_returns_reflection_result(self):
        result, trajectory = self.pipeline.get_extract_result(
            text="Alice is an engineer. Alice is Bob's spouse.",
            show_trajectory=True,
        )
        self.assertEqual(result["review"]["overall_score"], 88)
        self.assertEqual(result["review"]["schema_compliance"], 92)
        self.assertEqual(result["review"]["major_problems"], ["invalid relation"])
        self.assertEqual(result["document"]["triples"][0]["Relation"], "spouse of")
        self.assertEqual(result["document"]["attributes"]["Alice"], ["profession: engineer"])
        self.assertIn("prepare", trajectory["trajectory"])
        self.assertIn("extraction_agent", trajectory["trajectory"])
        self.assertIn("reflection_agent", trajectory["trajectory"])

    def test_pipeline_supports_file_input(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as file:
            file.write("Alice is an engineer. Alice is Bob's spouse.")
            file_path = file.name

        try:
            result, _ = self.pipeline.get_extract_result(
                use_file=True,
                file_path=file_path,
            )
            self.assertEqual(result["document"]["entities"][0]["name"], "Alice")
        finally:
            os.remove(file_path)

    def test_pipeline_validates_text_or_file(self):
        with self.assertRaises(ValueError):
            self.pipeline.get_extract_result()

    def test_pipeline_can_skip_reflection(self):
        result, trajectory = self.pipeline.get_extract_result(
            text="Alice is an engineer. Alice is Bob's spouse.",
            use_reflection=False,
        )
        self.assertTrue(result["review"]["skipped"])
        self.assertTrue(trajectory["trajectory"]["reflection_agent"]["skipped"])
        self.assertEqual(result["document"]["triples"][0]["Relation"], "spouse of")

    def test_pipeline_generates_text_from_triplets(self):
        result, trajectory = self.pipeline.get_generation_result(
            triplets=[
                {"subject": "The Legend of the Golden Gun", "relation": "publication date", "object": "1979"},
                {"subject": "The Sacketts", "relation": "publication date", "object": "1979"},
                {"subject": "The Shadow Riders (film)", "relation": "screenwriter", "object": "Louis L'Amour"},
                {"subject": "Louis L'Amour", "relation": "genre", "object": "Western (genre)"},
            ],
            show_trajectory=True,
        )
        self.assertIn("published in 1979", result["text"])
        self.assertEqual(result["triples"][0]["subject"], "The Legend of the Golden Gun")
        self.assertIn("reverse_generation_agent", trajectory["trajectory"])

    def test_load_config_supports_task_and_generation_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
            file.write(
                """
task:
  type: generation
  use_reflection: false

model:
  model_name_or_path: fake-model
  api_key: fake-key
  base_url: https://api.openai.com/v1

generation:
  triplets_json:
    - subject: Alice
      relation: profession
      object: engineer
  show_trajectory: true
                """.strip()
            )
            yaml_path = file.name

        try:
            config = load_extraction_config(yaml_path)
            self.assertEqual(config["task"]["type"], "generation")
            self.assertFalse(config["task"]["use_reflection"])
            self.assertEqual(config["generation"]["triplets_json"][0]["subject"], "Alice")
        finally:
            os.remove(yaml_path)

    def test_load_config_supports_generation_json_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as json_file:
            json_file.write(
                """
[
  {
    "subject": "Alice",
    "relation": "profession",
    "object": "engineer"
  }
]
                """.strip()
            )
            json_path = json_file.name

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as yaml_file:
            yaml_file.write(
                f"""
task:
  type: generation
  use_reflection: false

model:
  model_name_or_path: fake-model
  api_key: fake-key
  base_url: https://api.openai.com/v1

generation:
  triplets_json_path: "{json_path}"
  show_trajectory: true
                """.strip()
            )
            yaml_path = yaml_file.name

        try:
            config = load_extraction_config(yaml_path)
            self.assertEqual(config["generation"]["triplets_json"][0]["subject"], "Alice")
            self.assertEqual(config["generation"]["triplets_json"][0]["relation"], "profession")
        finally:
            os.remove(json_path)
            os.remove(yaml_path)

    def test_extract_json_dict_keeps_valid_json_with_single_quotes_in_strings(self):
        raw_response = """
{
  "schema_compliance": 95,
  "faithfulness": 90,
  "completeness": 75,
  "normalization": 85,
  "overall_score": 86,
  "major_problems": [
    "Faithfulness Error: The triple 'LLM' -> 'used for' -> '归纳' is misleading.",
    "Faithfulness Error: The triple 'SFT' -> 'used for' -> '减少模型大小和训练成本' is inaccurate."
  ],
  "minor_problems": [
    "Relation Redundancy: Multiple triples use 'used for' for different contexts."
  ]
}
        """.strip()

        parsed = extract_json_dict(raw_response)

        self.assertEqual(parsed["overall_score"], 86)
        self.assertIn("'LLM' -> 'used for' -> '归纳'", parsed["major_problems"][0])


if __name__ == "__main__":
    unittest.main()

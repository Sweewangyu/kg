import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.llm_def import BaseEngine
from pipeline import Pipeline
from utils.config_manager import ConfigManager


class FakeEngine(BaseEngine):
    def __init__(self):
        super().__init__("fake")

    def get_chat_response(self, prompt: str) -> str:
        if "Extraction Result To Review" in prompt:
            return """
            {
              "score": 88,
              "problems": ["invalid relation", "invalid attribute"],
              "suggestions": ["remove unsupported labels"],
              "revised_json": {
                "entities": [
                  {"name": "Alice", "type": "person"},
                  {"name": "Bob", "type": "person"}
                ],
                "attributes": {
                  "Alice": ["profession: engineer", "foo: value"]
                },
                "triples": [
                  {"Head": "Alice", "Relation": "spouse of", "Tail": "Bob"},
                  {"Head": "Alice", "Relation": "partner of", "Tail": "Bob"}
                ]
              }
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
        self.assertEqual(result["score"], 88)
        self.assertEqual(result["revised_json"]["triples"][0]["Relation"], "spouse of")
        self.assertEqual(result["revised_json"]["attributes"]["Alice"], ["profession: engineer"])
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
            self.assertEqual(result["revised_json"]["entities"][0]["name"], "Alice")
        finally:
            os.remove(file_path)

    def test_pipeline_validates_text_or_file(self):
        with self.assertRaises(ValueError):
            self.pipeline.get_extract_result()


if __name__ == "__main__":
    unittest.main()

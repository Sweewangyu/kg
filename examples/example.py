import sys

sys.path.append("./src")

from models.llm_def import OpenAIModel
from pipeline import Pipeline

model = OpenAIModel(model_name_or_path="your_model_name_or_path", api_key="your_api_key")
pipeline = Pipeline(model)

text = "Doctors Without Borders treated wounded people at Donka Hospital in Conakry, the capital of Guinea."

result, trajectory = pipeline.get_extract_result(
    text=text,
    show_trajectory=True,
)

generated_result, generation_trajectory = pipeline.get_generation_result(
    triplets=[
        {"subject": "The Legend of the Golden Gun", "relation": "publication date", "object": "1979"},
        {"subject": "The Sacketts", "relation": "publication date", "object": "1979"},
    ],
    show_trajectory=True,
)

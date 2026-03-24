import sys
sys.path.append("./src")
from models.llm_def import OpenAIModel
from dataset_def import REDataset

data_dir = "./data/datasets/NYT11/"
model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
api_key = sys.argv[2] if len(sys.argv) > 2 else ""
base_url = sys.argv[3] if len(sys.argv) > 3 else "https://api.openai.com/v1"
model = OpenAIModel(model_name_or_path=model_name, api_key=api_key, base_url=base_url)
dataset = REDataset(name="NYT11", data_dir=data_dir)
f1_score = dataset.evaluate(llm=model, mode="quick")
print("f1_score: ", f1_score)

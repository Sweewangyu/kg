import sys
sys.path.append("./src")
from models.llm_def import OpenAIModel
from dataset_def import NERDataset
name = "crossner-"
data_dir = "./data/datasets/CrossNER/"
model = OpenAIModel(model_name_or_path="gpt-4o-mini", api_key="your_api_key", base_url="https://api.openai.com/v1")
tasklist = ["ai", "literature", "music", "politics", "science"]
for task in tasklist:
    task_name = name + task
    task_data_dir = data_dir + task
    dataset = NERDataset(name=task_name, data_dir=task_data_dir)
    mode = "quick"
    f1_score = dataset.evaluate(llm=model, mode=mode)
    print(f"Task: {task_name}, f1_score: {f1_score}")

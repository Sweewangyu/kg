# KG-Extraction Framework

## 项目简介 (Introduction)
本项目是一个基于大语言模型 (LLMs) 的信息抽取 (Information Extraction) 和知识图谱构建 (Knowledge Graph Construction) 框架。它支持多种主流信息抽取任务，能够从非结构化文本中抽取出结构化信息，并且支持直接将结果构建为 Neo4j 知识图谱。

## 核心功能 (Core Features)
- **多任务支持**：支持命名实体识别 (NER)、关系抽取 (RE)、事件抽取 (EE) 以及三元组抽取 (Triple)。
- **多Agent协作**：基于 SchemaAgent、ExtractionAgent 和 ReflectionAgent 的协作机制，提供多种抽取模式 (`quick`, `standard`, `customized`)。
- **知识图谱构建**：支持将抽取的结构化数据转化为 Cypher 语句，直接写入 Neo4j 数据库生成知识图谱。
- **灵活的输入方式**：支持直接输入文本字符串，或者从本地文件读取（原生支持 TXT、PDF、HTML 格式）。
- **可配置化驱动**：通过 YAML 配置文件进行快速任务定义和运行，同时支持灵活的 Python API 编程调用。

## 目录结构 (Directory Structure)
- `data/`: 包含各种数据集文件 (如 CrossNER, NYT11) 以及供测试的输入文档。
- `examples/`: 包含不同抽取任务的 YAML 配置文件示例和 Python API 调用脚本示例。
- `experiments/`: 用于模型和数据集测试评估的实验脚本。
- `figs/`: 文档和介绍所用的图片。
- `src/`: 核心源代码。
  - `construct/`: 包含将抽取结果转换为图数据库 Cypher 语句的逻辑。
  - `models/`: LLM 接口封装与提示词定义（支持所有 OpenAI 格式的 API，如 ChatGPT, DeepSeek 等）。
  - `modules/`: 包含各类 Agent (Schema, Extraction, Reflection) 和知识库样例管理逻辑。
  - `utils/`: 数据结构定义及文本预处理等工具类。
  - `config.yaml`: 全局默认抽取模式和 Prompt 配置文件。
  - `pipeline.py`: 定义信息抽取和图谱构建流水线的核心逻辑。
  - `run.py`: 框架命令行执行入口。

## 环境安装 (Installation)
1. 确保已安装 Python (推荐 3.8+)。
2. 克隆本项目后，在项目根目录执行以下命令安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 若需要将抽取结果自动构建为知识图谱，请确保已在本地或服务器安装并启动 Neo4j 数据库。

## 使用方法 (Usage)

### 方式一：基于命令行的 YAML 配置文件运行 (推荐)
可以通过修改 `examples/config` 目录下的 YAML 配置文件，运行 `src/run.py` 进行抽取任务。
例如，运行一个 NER（命名实体识别）任务：
```bash
python src/run.py --config examples/config/NER.yaml
```

**YAML 配置文件详解**：
```yaml
model:
  category: OpenAIModel  # 模型类别（需与 src/models/llm_def.py 中的类名对应，如 OpenAIModel）
  model_name_or_path: gpt-4o-mini # 具体的模型名称
  api_key: "your_api_key" # LLM API Key
  base_url: "https://api.openai.com/v1" # API 服务地址

extraction:
  task: NER  # 任务类型，支持 Base, NER, RE, EE, Triple
  text: "这里是需要抽取的文本内容..." # 文本输入（若 use_file 为 false）
  constraint: ["person", "location", "organization"] # 约束条件（如指定要抽取的实体/关系/事件类型）
  use_file: false # 是否使用文件输入
  file_path: "" # 若使用文件，此处填写文件路径 (支持 .txt, .pdf, .html)
  mode: quick # 抽取模式，可选 quick, standard, customized (详见 src/config.yaml)
  update_case: false # 是否将本次结果作为样例更新到历史 Case 知识库
  show_trajectory: true # 是否在控制台打印 Agent 的中间推理轨迹

# construct: # (可选) 若需要构建 Neo4j 知识图谱则取消注释并填写以下字段
#   database: Neo4j
#   url: neo4j://localhost:7687
#   username: your_username
#   password: your_password
```
*(注：请确保 config 中的 `category` 与 `src/models/llm_def.py` 中的类名一致。)*

### 方式二：基于 Python API 编程调用
可以通过实例化 `Pipeline` 直接在代码中灵活调用，具体参考 `examples/example.py`：
```python
import sys
sys.path.append("./src")
from models import OpenAIModel
from pipeline import Pipeline

# 1. 初始化模型配置
model = OpenAIModel(model_name_or_path="gpt-4o-mini", api_key="your_api_key")
pipeline = Pipeline(model)

# 2. 定义抽取任务配置
Task = "NER"
Text = "Finally , every other year , ELRA organizes a major conference LREC..."
Constraint = ["conference", "organization", "location", "person"]

# 3. 获取抽取结果
result, trajectory = pipeline.get_extract_result(
    task=Task, 
    text=Text, 
    constraint=Constraint, 
    show_trajectory=True
)
print("抽取结果:\n", result)
```

## 高级功能 (Advanced Features)
- **多模式抽取策略** (`mode`)：
  - `quick`: 快速模式，直接通过 SchemaAgent 推导后进行信息抽取，速度最快。
  - `standard`: 标准模式，结合 Case Repository (历史样例库) 进行 Few-Shot 抽取，并通过 ReflectionAgent 对初步抽取结果进行反思与自动修正。
  - `customized`: 自定义模式，基于检索到的特定 Schema 自定义抽取逻辑。
- **动态知识库更新与自我迭代** (`update_case: true`)：允许在抽取过程中由人工介入提供或确认标准答案 (Ground Truth)。系统会将其作为优质案例动态保存到 `case_repository.json` 中，显著提升后续同类任务的 In-Context Learning 准确性。
- **直接构建知识图谱**：提供 `Triple` 任务并结合 `construct` 配置，能够一键完成“无结构文本 -> 三元组提取 -> 自动生成 Cypher -> 图数据库入库”的全链路知识图谱构建流程。
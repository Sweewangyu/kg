# KG-Extraction Framework

## 项目简介
本项目已裁剪为一个只保留知识图谱联合抽取的轻量框架。整体流程聚焦于“输入准备 → extraction_agent 联合抽取 → reflection_agent 质量反思与修正”，不再包含 RE、Triple、多阶段拆分、任务策略分支、知识图谱入库、实验评测与历史兼容逻辑。

## 核心能力
- 只支持 KG 单任务
- 支持直接输入文本或读取本地 TXT、PDF 文件
- 统一通过 OpenAI 兼容接口接入大模型
- 支持 YAML 配置与 Python API 两种调用方式
- 支持打印主流程轨迹，便于调试和测试

## 目录结构
- `examples/config/`: KG 的最小可运行配置
- `examples/example.py`: Python API 示例
- `src/models/`: 大模型封装
- `src/modules/`: extraction_agent 与 reflection_agent
- `src/utils/`: 配置读取、数据结构与 JSON 清洗
- `src/prompt.py`: Prompt、默认 schema、结构校验与结果规范化
- `src/pipeline.py`: 主干流水线
- `src/run.py`: 命令行入口

## 安装
```bash
pip install -r requirements.txt
```

## 命令行运行
```bash
python src/run.py --config examples/config/KG.yaml
```

## YAML 配置
```yaml
model:
  model_name_or_path: gpt-4o-mini
  api_key: your_api_key
  base_url: https://api.openai.com/v1

extraction:
  text: 这里填写待抽取文本
  use_file: false
  file_path: ""
  language: auto
  show_trajectory: true
```

## Python API
```python
import sys

sys.path.append("./src")

from models.llm_def import OpenAIModel
from pipeline import Pipeline

model = OpenAIModel(
    model_name_or_path="your_model_name_or_path",
    api_key="your_api_key",
    base_url="https://api.openai.com/v1",
)
pipeline = Pipeline(model)

result, trajectory = pipeline.get_extract_result(
    text="Doctors Without Borders treated wounded people at Donka Hospital in Conakry, the capital of Guinea.",
    show_trajectory=True,
)
print(result)
```

## 设计说明
- `Pipeline` 负责单轮编排与最终结果输出
- `ExtractionAgent` 负责一次性抽取 entities、attributes、triples
- `ReflectionAgent` 负责评分、发现问题并返回 revised_json
- `prompt.py` 负责 schema 归一化、结构校验与 schema 合规过滤

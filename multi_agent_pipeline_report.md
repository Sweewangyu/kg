# Multi-Agent 信息抽取 Pipeline 技术报告

本报告详细解析了基于 Multi-Agent 架构的信息抽取流水线（Pipeline）的完整执行逻辑。该系统通过定义统一的数据总线 `DataPoint`，将核心业务逻辑拆解到 `SchemaAgent`、`ExtractionAgent` 和 `ReflectionAgent` 三个智能体中，形成“模式定义 -> 信息抽取 -> 自省优化”的闭环机制。此外，系统集成了嵌入模型（Embedding Model）和案例库（Case Repository）来实现检索增强，并支持将抽取结果直接构建为知识图谱（Knowledge Graph）。

---

## 1. 系统架构与调用链

### 1.1 数据总线：`DataPoint`
在整个流水线中，数据不再是零散的变量，而是封装在 `DataPoint` 对象中。各个 Agent 接收 `DataPoint`，修改其内部状态并返回。
```python
# src/utils/data_def.py
class DataPoint:
    def __init__(self, task: TaskType = "Base", instruction: str = "", text: str = "", ...):
        # 初始输入
        self.task = task
        self.instruction = instruction
        self.text = text
        self.output_schema = output_schema
        self.constraint = constraint
        
        # 中间状态（各 Agent 逐步填充）
        self.chunk_text_list = []    # 分块后的文本
        self.print_schema = ""       # 生成的 Schema 代码或结构
        self.distilled_text = ""     # 文本领域/体裁提炼
        self.result_list = []        # 分块抽取结果
        
        # 最终输出与日志
        self.pred = ""               # 最终汇总的抽取结果
        self.result_trajectory = {}  # 记录所有 Agent 执行的历史轨迹
```

### 1.2 Pipeline 调用链逻辑
`Pipeline.get_extract_result` 是整个系统的入口，其调用链由配置文件 `config.yaml` 中的 `mode` 动态决定（例如 `quick`、`standard`、`customized` 模式）。经过架构重构，该方法采用高内聚低耦合的私有方法进行流水线编排，并支持知识图谱构建和案例库更新。

**核心代码执行链：**
```python
# src/pipeline.py - get_extract_result 方法节选

# 1. 构建基础数据对象 DataPoint
data = self.__build_data(task, instruction, text, output_schema, constraint, use_file, file_path, truth)

# 2. 解析执行模式，组装方法链 (例如：{'schema_agent': 'get_deduced_schema', ...})
sorted_process_method = self.__resolve_process_method(data, mode)

# 3. 将方法链实例化为可执行的步骤 (AgentStep 模式)
steps = self.__build_steps(sorted_process_method)

# 4. 依次执行所有 Agent 步骤，并对结果进行最终汇总
data = self.__run_steps(data, steps)

# 5. 可选：构建知识图谱 (Knowledge Graph)
if iskg:
    self.__construct_kg(extraction_result, construct)

# 6. 可选：更新案例库 (Case Repository)
if update_case:
    self.__update_case(data)
```

---

## 2. 嵌入模型 (Embedding Model) 与案例库机制

系统在 `CaseRepositoryHandler` 中深度集成了嵌入模型，使其扮演**语义搜索引擎**的角色，为抽取和自省提供高质量的上下文。

### 2.1 嵌入模型选型与配置
- **模型名称**：系统默认使用轻量级、高性能的 `all-MiniLM-L6-v2`。
- **配置路径**：位于 `src/config.yaml` 中的 `model.embedding_model`。
- **作用域**：主要用于全局语义检索和案例的向量化存储。

### 2.2 混合检索算法 (Hybrid Retrieval)
为了保证 Few-shot 案例的相关性，系统采用了向量空间检索与传统文本匹配相结合的方案：
1. **语义匹配 (Semantic Similarity)**：利用嵌入模型将 Query 和案例库文本编码为高维向量，计算余弦相似度，捕捉深层语义关联，解决“语义近但字面远”的问题。
2. **字符串匹配 (String Similarity)**：结合模糊字符串匹配算法（如 Fuzzy Matching），确保术语和关键词的精确对齐。
3. **加权评分**：系统对两种分数进行最大最小归一化处理，综合评估后选出最相关的 Top-K 案例注入 Prompt。

### 2.3 案例的动态迭代与作用
- **正向案例 (Good Case)**：在 `ExtractionAgent` 中被检索和注入，引导 LLM 模仿正确的抽取格式和逻辑。
- **负向案例 (Bad Case)**：在 `ReflectionAgent` 中使用，作为反面教材提醒 LLM 避开常见的抽取陷阱（如误将动作识别为实体）。
- **动态进化**：当流水线启用 `update_case=True` 时，用户确认的正确案例会被嵌入模型实时编码，追加到向量索引中，实现系统的自我进化。

---

## 3. 核心智能体详解与 Prompt 机制

以下展示以中文 Prompt 为例（系统内使用了 `BilingualPromptTemplate` 支持双语）。

### 3.1 SchemaAgent (模式定义者)
**职责**：基于用户任务指令和具体文本，通过 LLM 动态推导并定义抽取目标的数据结构（Schema）。
- **输入**：原始文本 `text`、任务指令 `instruction`。
- **输出**：更新 `DataPoint.output_schema`（Pydantic 结构或 JSON 规范）以及 `DataPoint.distilled_text`（文本特征）。

**执行链**：
1. **文本分析 (Text Analysis)**：提取文本所属领域。
   ```text
   **指令**：请对给定文本进行分析和分类。
   {examples}
   **文本**：{text}
   **输出格式**：{schema}
   ```
2. **Schema 推导 (Deduced Schema)**：基于分析结果和文本，生成结构化代码。
   ```text
   **指令**：根据提供的文本和任务描述，使用 Pydantic 定义 Python 输出结构。将最终的提取目标类命名为 'ExtractionTarget'。
   {examples}
   **任务**：{instruction}
   **文本**：{distilled_text}
   {text}
   现在请推导输出结构。确保输出的代码片段被 '```' 包裹，并且可以直接被 Python 解释器解析。
   **输出结构**： 
   ```

### 3.2 ExtractionAgent (信息抽取者)
**职责**：根据 SchemaAgent 推导出的结构，结合案例库（Case Repository）通过嵌入模型检索出的正例（Few-shot），执行具体的抽取任务。
- **输入**：`DataPoint` 中的 `output_schema`、`constraint`（实体/关系约束）和 `chunk_text_list`。
- **输出**：向 `DataPoint.result_list` 填充分块抽取的结果字典。

**核心 Prompt**：
```text
**指令**：你是一个擅长信息提取的智能体。{instruction}
{examples}  # 这里会由 CaseRepositoryHandler 结合嵌入模型注入高质量的正向抽取案例

**文本**：{text}
{additional_info} # 这里会注入约束条件（如只允许特定类型的实体）

**输出格式**：{schema}

现在请从文本中提取相应的信息。确保你提取的信息在给定文本中有明确的引用。将文本中未明确提及的任何属性设置为 null。
```

### 3.3 ReflectionAgent (自省与优化者)
**职责**：引入“自我一致性（Self-Consistency）”检验，通过多次重采样发现歧义或错误。若有冲突，则引入案例库中检索出的反例（Bad Case）进行深度反思（Reflection）与纠错。
- **输入**：初步抽取出的 `result_list`。
- **输出**：覆盖原始结果，更新为纠错后的 `result_list`。

**核心自省代码逻辑**：
```python
# src/modules/reflection_agent.py
# 修改温度参数 (0.5, 1) 进行重采样
for index in range(2):
    self.module.llm.set_hyperparameter(temperature=temperature[index])
    data = extract_func(data)
    result_trails.append(data.result_list)
# 投票机制：对比 3 次结果，若有分歧，将该索引标记为 reflect_index 进行下一步深度反思。
```

**反思纠错 Prompt**：
```text
**指令**：你是一个擅长基于原始结果进行反思和优化的智能体。请参考**反思参考**来识别当前提取结果中的潜在问题。

**反思参考**：{examples} # 这里注入 Bad Case 及其 Reflection

现在请审查提取结果中的每个元素。根据反思识别并改进结果中的任何潜在问题。注意：如果原始结果是正确的，则不需要进行任何修改！

**任务**：{instruction}
**文本**：{text}
**输出格式**：{schema}
**原始结果**：{result} # 存在分歧的原始结果
```

---

## 4. 完整案例推演 (Concrete Example)

假设我们要执行一个医疗领域的命名实体识别（NER）任务。
- **原始输入**：`task="NER"`, `text="患者张三，45岁，于2023年3月15日被确诊为2型糖尿病。"`, `mode="standard"`

### 【步骤 1】初始化
- `Pipeline` 生成 `DataPoint`，自动匹配指令：`instruction="Extract the Named Entities in the given text."`。
- 排定执行链：`SchemaAgent -> ExtractionAgent -> ReflectionAgent`。

### 【步骤 2】SchemaAgent 执行
1. **输入**：`text="患者张三..."`
2. **文本分析**：大模型识别出 `field="医疗健康"`, `genre="病历诊断"`。
3. **Schema 生成**：注入文本和指令，LLM 返回一段 Pydantic 代码片段，定义了包含 `患者姓名`、`年龄`、`确诊日期`、`疾病名称` 等字段的 `ExtractionTarget` 类。
4. **输出**：将这段代码规范序列化并保存到 `DataPoint.output_schema` 中。

### 【步骤 3】ExtractionAgent 执行
1. **输入**：上一步的 `output_schema`，原始 `text`。
2. **案例检索**：基于嵌入模型从案例库计算余弦相似度，查询出“医疗健康实体抽取”的正例（Good Case）注入 prompt。
3. **初步抽取**：LLM 生成初步结果：
   ```json
   {
     "entity_list": [
       {"name": "张三", "type": "患者姓名"},
       {"name": "45岁", "type": "年龄"},
       {"name": "确诊", "type": "动词"}  // <--- 这是一个错误实体
     ]
   }
   ```
4. **输出**：初步结果存入 `DataPoint.result_list[0]`。

### 【步骤 4】ReflectionAgent 执行
1. **自我一致性检查**：系统将 Temperature 设为 0.5 和 1 重跑两次 ExtractionAgent。发现某次重跑没有提取出“确诊”这个实体，产生分歧。
2. **触发反思**：基于嵌入模型检索到高度相关的 Bad Case（提示：“不要将动作行为作为实体提取”）。
3. **纠错执行**：将包含“确诊”的 `原始结果` 连同 Bad Case 传入 `REFLECT_INSTRUCTION_CN`。
4. **输出修正结果**：LLM 意识到错误，剔除“确诊”实体，输出最终的高质量 JSON：
   ```json
   {
     "entity_list": [
       {"name": "张三", "type": "患者姓名"},
       {"name": "45岁", "type": "年龄"},
       {"name": "2023年3月15日", "type": "确诊日期"},
       {"name": "2型糖尿病", "type": "疾病名称"}
     ]
   }
   ```

### 【步骤 5】后续处理与返回
- 由于文本较短只分了一个 Chunk，`ExtractionAgent.summarize_answer` 直接汇总。
- **图谱构建 (可选)**：如果配置了 `iskg=True`，系统会将 JSON 结果转换为 Cypher 语句，并自动写入 Neo4j 数据库。
- **案例更新 (可选)**：如果启用了 `update_case`，经过确认的正确抽取结果会被嵌入模型编码并追加到案例库，优化下一次检索。
- 最终结果与完整交互记录（`trajectory`）打包返回。

---

## 5. 评测机制 (Evaluation Mechanism)

在 `experiments` 目录下，系统提供了针对特定任务（如 NER, RE 等）的标准化评测机制。通过继承 `BaseDataset` 并重写 `evaluate` 方法，可以对 Pipeline 的抽取效果进行量化评估。

### 5.1 数据集封装与调用
以命名实体识别（NER）为例，`NERDataset` 类封装了测试集数据加载、Schema 读取以及批量预测逻辑。
```python
# experiments/run_ner.py 示例
dataset = NERDataset(name="crossner-ai", data_dir="./data/datasets/CrossNER/ai")
f1_score = dataset.evaluate(llm=model, mode="quick")
```

### 5.2 `evaluate` 方法执行原理
`evaluate` 核心方法执行了以下关键步骤：

1. **样本控制与输出初始化**：
   - 支持全量或按需采样（`sample` 和 `random_sample`），减少测试等待时间。
   - 定义 `.jsonl` 格式的输出文件，保存每条测试记录及其详细轨迹。

2. **批量推理与重试机制**：
   - 遍历测试样本，调用 `Pipeline.get_extract_result()` 获取预测结果。
   - **重试机制**：内置 `self.retry`（默认重试 2 次），以应对偶发的 LLM 结果解析失败（如返回非标 JSON 等异常）。

3. **结果对齐与指标计算**：
   - **格式归一化**：预测结果与真实标签（Truth）都通过 `dict_list_to_set` 方法转换为**无序集合（Set）**。此举消除了列表顺序差异带来的误差。
   - **计算指标**：调用 `calculate_metrics(truth_set, pred_set)` 计算出真正的 TP（True Positive）、FP（False Positive）和 FN（False Negative），进而计算得出该样本的 `Precision`（精确率）、`Recall`（召回率）以及 `F1 Score`。

4. **结果持久化与汇总**：
   - 每预测完一条样本，立即将原始预测、真实标签、轨迹明细和当前样本指标以 JSONL 格式追加写入文件。
   - 所有样本处理完成后，计算并输出**全局平均 Precision、Recall、F1 Score**，并追加在日志文件末尾，形成直观的最终测试报告。

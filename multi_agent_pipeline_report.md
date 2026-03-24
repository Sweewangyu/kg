# Multi-Agent 信息抽取 Pipeline 技术报告

本报告详细解析了基于 Multi-Agent 架构的信息抽取流水线（Pipeline）的完整执行逻辑。该系统通过定义统一的数据总线 `DataPoint`，将核心业务逻辑拆解到 `SchemaAgent`、`ExtractionAgent` 和 `ReflectionAgent` 三个智能体中，形成“模式定义 -> 信息抽取 -> 自省优化”的闭环机制。

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
`Pipeline.get_extract_result` 是整个系统的入口，其调用链由配置文件 `config.yaml` 中的 `mode` 动态决定（例如 `standard` 模式包含完整的三步）。

**核心代码执行链：**
```python
# src/pipeline.py - get_extract_result 方法节选

# 1. 组装方法链 (例如：{'schema_agent': 'get_deduced_schema', 'extraction_agent': 'extract_information_with_case', ...})
sorted_process_method = self.__init_method(data, process_method)

# 2. 依次调用各个 Agent 模块
for agent_name, method_name in sorted_process_method.items():
    agent = getattr(self, agent_name, None)
    method = getattr(agent, method_name, None)
    
    # 将 data 传入下一个 agent，更新内部状态
    data = method(data)  

# 3. 最终汇总与收尾
data = self.extraction_agent.summarize_answer(data)
```

---

## 2. 核心智能体详解与 Prompt 机制

以下展示以中文 Prompt 为例（系统内使用了 `BilingualPromptTemplate` 支持双语）。

### 2.1 SchemaAgent (模式定义者)
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

### 2.2 ExtractionAgent (信息抽取者)
**职责**：根据 SchemaAgent 推导出的结构，结合案例库（Case Repository）中的正例（Few-shot），执行具体的抽取任务。
- **输入**：`DataPoint` 中的 `output_schema`、`constraint`（实体/关系约束）和 `chunk_text_list`。
- **输出**：向 `DataPoint.result_list` 填充分块抽取的结果字典。

**核心 Prompt**：
```text
**指令**：你是一个擅长信息提取的智能体。{instruction}
{examples}  # 这里会由 CaseRepositoryHandler 注入高质量的正向抽取案例

**文本**：{text}
{additional_info} # 这里会注入约束条件（如只允许特定类型的实体）

**输出格式**：{schema}

现在请从文本中提取相应的信息。确保你提取的信息在给定文本中有明确的引用。将文本中未明确提及的任何属性设置为 null。
```

### 2.3 ReflectionAgent (自省与优化者)
**职责**：引入“自我一致性（Self-Consistency）”检验，通过多次重采样发现歧义或错误。若有冲突，则引入案例库中的反例（Bad Case）进行深度反思（Reflection）与纠错。
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

## 3. 完整案例推演 (Concrete Example)

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
2. **案例检索**：从案例库查询“医疗健康实体抽取”的正例（Good Case）注入 prompt。
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
2. **触发反思**：检索到 Bad Case（提示：“不要将动作行为作为实体提取”）。
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

### 【步骤 5】Summarize 与返回
由于文本较短只分了一个 Chunk，`ExtractionAgent.summarize_answer` 直接返回最终结果。最终结果与完整交互记录（`trajectory`）打包，返回给调用方。

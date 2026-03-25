import re

PROMPT_TEMPLATES = {
    "en": {
        "text_analysis": """
**Instruction**: Please analyze and categorize the given text.
{examples}
**Text**: {text}

**Output Schema**: {schema}
""",
        "deduced_schema_json": """
**Instruction**: Generate an output format that meets the requirements as described in the task. Pay attention to the following requirements:
    - Format: Return your responses in dictionary format as a JSON object.
    - Content: Do not include any actual data; all attribute values should be set to None.
    - Note: Attributes not mentioned in the task description should be ignored.
{examples}
**Task**: {instruction}

**Text**: {distilled_text}
{text}

Now please deduce the output schema in JSON format. All attribute values should be set to None.
**Output Schema**:
""",
        "deduced_schema_code": """
**Instruction**: Based on the provided text and task description, define the output schema in Python using Pydantic. Name the final extraction target class as 'ExtractionTarget'.
{examples}
**Task**: {instruction}

**Text**: {distilled_text}
{text}

Now please deduce the output schema. Ensure that the output code snippet is wrapped in '```' and can be directly parsed by the Python interpreter.
**Output Schema**:
""",
        "extract": """
**Instruction**: You are an agent skilled in information extraction. {instruction}
{examples}
**Text**: {text}
{constraint}
**Output Schema**: {schema}

Now please extract the corresponding information from the text. Ensure that the information you extract has a clear reference in the given text. Set any property not explicitly mentioned in the text to null.
""",
        "reflect": """
**Instruction**: You are an agent skilled in reflection and optimization based on the original result. Refer to **Reflection Reference** to identify potential issues in the current extraction results.

**Reflection Reference**: {examples}

Now please review each element in the extraction result. Identify and improve any potential issues in the result based on the reflection. NOTE: If the original result is correct, no modifications are needed.

**Task**: {instruction}

**Text**: {text}

**Output Schema**: {schema}

**Original Result**: {result}
""",
        "summarize": """
**Instruction**: Below is a list of results obtained after segmenting and extracting information from a long article. Please consolidate all the answers to generate a final response.

**Task**: {instruction}

**Result List**: {answer_list}
{constraint}
**Output Schema**: {schema}

Now summarize the information from the Result List.
""",
        "good_case_analysis": """
**Instruction**: Below is an information extraction task and its corresponding correct answer. Provide the reasoning steps that led to the correct answer, along with a brief explanation of the answer. Your response should be brief and organized.

**Task**: {instruction}

**Text**: {text}
{additional_info}
**Correct Answer**: {result}

Now please generate the reasoning steps and brief analysis of the **Correct Answer** given above. DO NOT generate your own extraction result.
**Analysis**:
""",
        "bad_case_reflection": """
**Instruction**: Based on the task description, compare the original answer with the correct one. Your output should be a brief reflection or concise summarized rules.

**Task**: {instruction}

**Text**: {text}
{additional_info}
**Original Answer**: {original_answer}

**Correct Answer**: {correct_answer}

Now please generate a brief and organized reflection. DO NOT generate your own extraction result.
**Reflection**:
""",
        "default_schema": "The final extraction result should be formatted as a JSON object.",
        "default_ner": "Extract the named entities in the given text.",
        "default_re": "Extract relationships between named entities in the given text.",
        "default_ee": "Extract the events in the given text.",
        "default_triple": "Extract the triples (subject, relation, object) from the given text, and try to cover all relationships for each entity.",
        "schema_serialized": """{schema_content}

For example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}}}, the object {{"foo": ["bar", "baz"]}} is a well-formatted instance.""",
        "text_analysis_summary": "This text is from the field of {field} and represents the genre of {genre}.",
        "partial_text": """Below is a portion of the text to be extracted.
{text}""",
        "schema_with_default": """{default_schema}
{schema}""",
        "constraint_ner": """
**Entity Type Constraint**: The type of entities must be chosen from the following list.
{constraint}
""",
        "constraint_re": """
**Relation Type Constraint**: The type of relations must be chosen from the following list.
{constraint}
""",
        "constraint_ee": """
**Event Extraction Constraint**: The event type must be selected from the following dictionary keys, and its event arguments should be chosen from its corresponding dictionary values.
{constraint}
""",
        "constraint_triple_entity": """
**Triple Extraction Constraint**: Entity types must be chosen from the following list.
{constraint}
""",
        "constraint_triple_relation": """
**Triple Extraction Constraint**: Relation types must be chosen from the following list.
{constraint}
""",
        "constraint_triple_ent_rel": """
**Triple Extraction Constraint**: Entity types must be chosen from the following list.
{ent_constraint}
Relation types must be chosen from the following list.
{rel_constraint}
""",
        "constraint_triple_rel_obj": """
**Triple Extraction Constraint**: Relation types must be chosen from the following list.
{rel_constraint}
Object entity types must be chosen from the following list.
{obj_constraint}
""",
        "constraint_triple_subj_obj": """
**Triple Extraction Constraint**: Subject entity types must be chosen from the following list.
{subj_constraint}
Object entity types must be chosen from the following list.
{obj_constraint}
""",
        "constraint_triple_subj_rel": """
**Triple Extraction Constraint**: Subject entity types must be chosen from the following list.
{subj_constraint}
Relation types must be chosen from the following list.
{rel_constraint}
""",
        "constraint_triple_all": """
**Triple Extraction Constraint**: Subject entity types must be chosen from the following list.
{subj_constraint}
Relation types must be chosen from the following list.
{rel_constraint}
Object entity types must be chosen from the following list.
{obj_constraint}
""",
        "constraint_triple_default": """
**Triple Extraction Constraint**: Entity types must be chosen from the following list.
{constraint}
""",
        "example_wrapper": """
Here are some examples:
{example}
(END OF EXAMPLES)
""",
        "good_case_wrapper": """
Here are some examples:
{example}
(END OF EXAMPLES)
Refer to the reasoning steps and analysis in the examples to help complete the extraction task below.
""",
        "bad_case_wrapper": """
Here are some examples of bad cases:
{example}
(END OF EXAMPLES)
Refer to the reflection rules and reflection steps in the examples to help optimize the original result below.
""",
        "case_text": """**Text**: {distilled_text}
{chunk_text}""",
        "case_task": "**Task**: {instruction}",
        "case_original_result": "**Original Result**: {pred}",
        "case_original_answer": "**Original Answer**: {pred}",
        "case_correct_answer": "**Correct Answer**: {truth}",
        "case_analysis": "**Analysis**: {analysis}",
        "case_reflection": "**Reflection**: {reflection}",
        "case_index_base": "**Task**: {instruction}",
        "case_index_bad_base": """**Task**: {instruction}

**Original Result**: {pred}""",
        "case_index_constraint": "{constraint}",
        "case_index_bad_constraint": """{constraint}

**Original Result**: {pred}""",
        "good_case_content_base": """{instruction}

{text}

{analysis}

{answer}""",
        "good_case_content_task": """{text}

{constraint}

{analysis}

{answer}""",
        "bad_case_content_base": """{instruction}

{text}

{original_answer}

{reflection}

{correct_answer}""",
        "bad_case_content_task": """{text}

{constraint}

{original_answer}

{reflection}

{correct_answer}"""
    },
    "zh": {
        "text_analysis": """
**指令**：请对给定文本进行分析和分类。
{examples}
**文本**：{text}

**输出格式**：{schema}
""",
        "deduced_schema_json": """
**指令**：根据任务描述生成符合要求的输出格式。请注意以下要求：
    - 格式：以 JSON 对象返回响应。
    - 内容：不要包含任何实际数据；所有属性值应设置为 None。
    - 注意：忽略任务描述中未提及的属性。
{examples}
**任务**：{instruction}

**文本**：{distilled_text}
{text}

现在请以 JSON 格式推导输出结构。所有属性值应设置为 None。
**输出格式**：
""",
        "deduced_schema_code": """
**指令**：根据提供的文本和任务描述，使用 Pydantic 定义 Python 输出结构。将最终的提取目标类命名为 'ExtractionTarget'。
{examples}
**任务**：{instruction}

**文本**：{distilled_text}
{text}

现在请推导输出结构。确保输出的代码片段被 '```' 包裹，并且可以直接被 Python 解释器解析。
**输出结构**：
""",
        "extract": """
**指令**：你是一个擅长信息提取的智能体。{instruction}
{examples}
**文本**：{text}
{constraint}
**输出格式**：{schema}

现在请从文本中提取相应的信息。确保你提取的信息在给定文本中有明确依据。将文本中未明确提及的任何属性设置为 null。
""",
        "reflect": """
**指令**：你是一个擅长基于原始结果进行反思和优化的智能体。请参考**反思参考**来识别当前提取结果中的潜在问题。

**反思参考**：{examples}

现在请审查提取结果中的每个元素。根据反思识别并改进结果中的潜在问题。注意：如果原始结果正确，则不需要进行任何修改。

**任务**：{instruction}

**文本**：{text}

**输出格式**：{schema}

**原始结果**：{result}
""",
        "summarize": """
**指令**：以下是从长篇文章中分段提取信息后获得的结果列表。请整合所有答案以生成最终响应。

**任务**：{instruction}

**结果列表**：{answer_list}
{constraint}
**输出格式**：{schema}

现在请总结结果列表中的信息。
""",
        "good_case_analysis": """
**指令**：以下是一个信息提取任务及其对应的正确答案。请提供得出正确答案的推理步骤，并对答案进行简要解释。你的回答应简明且有条理。

**任务**：{instruction}

**文本**：{text}
{additional_info}
**正确答案**：{result}

现在请生成对上述**正确答案**的推理步骤和简要分析。不要生成你自己的提取结果。
**分析**：
""",
        "bad_case_reflection": """
**指令**：根据任务描述，将原始答案与正确答案进行比较。你的输出应该是简短的反思或简明的总结规则。

**任务**：{instruction}

**文本**：{text}
{additional_info}
**原始答案**：{original_answer}

**正确答案**：{correct_answer}

现在请生成一个简明有条理的反思。不要生成你自己的提取结果。
**反思**：
""",
        "default_schema": "最终的提取结果应格式化为 JSON 对象。",
        "default_ner": "提取给定文本中的命名实体。",
        "default_re": "提取给定文本中命名实体之间的关系。",
        "default_ee": "提取给定文本中的事件。",
        "default_triple": "从给定文本中提取三元组（主语、关系、宾语），并尽量覆盖每个实体的全部关系。",
        "schema_serialized": """{schema_content}

例如，对于结构 {{"properties": {{"foo": {{"title": "Foo", "description": "字符串列表", "type": "array", "items": {{"type": "string"}}}}}}}}, 对象 {{"foo": ["bar", "baz"]}} 是一个格式良好的实例。""",
        "text_analysis_summary": "这段文本来自 {field} 领域，属于 {genre} 体裁。",
        "partial_text": """以下是待提取文本的一部分。
{text}""",
        "schema_with_default": """{default_schema}
{schema}""",
        "constraint_ner": """
**实体类型约束**：实体类型必须从以下列表中选择。
{constraint}
""",
        "constraint_re": """
**关系类型约束**：关系类型必须从以下列表中选择。
{constraint}
""",
        "constraint_ee": """
**事件提取约束**：事件类型必须从以下字典键中选择，其事件参数应从相应的字典值中选择。
{constraint}
""",
        "constraint_triple_entity": """
**三元组提取约束**：实体类型必须从以下列表中选择。
{constraint}
""",
        "constraint_triple_relation": """
**三元组提取约束**：关系类型必须从以下列表中选择。
{constraint}
""",
        "constraint_triple_ent_rel": """
**三元组提取约束**：实体类型必须从以下列表中选择。
{ent_constraint}
关系类型必须从以下列表中选择。
{rel_constraint}
""",
        "constraint_triple_rel_obj": """
**三元组提取约束**：关系类型必须从以下列表中选择。
{rel_constraint}
客体实体类型必须从以下列表中选择。
{obj_constraint}
""",
        "constraint_triple_subj_obj": """
**三元组提取约束**：主体实体类型必须从以下列表中选择。
{subj_constraint}
客体实体类型必须从以下列表中选择。
{obj_constraint}
""",
        "constraint_triple_subj_rel": """
**三元组提取约束**：主体实体类型必须从以下列表中选择。
{subj_constraint}
关系类型必须从以下列表中选择。
{rel_constraint}
""",
        "constraint_triple_all": """
**三元组提取约束**：主体实体类型必须从以下列表中选择。
{subj_constraint}
关系类型必须从以下列表中选择。
{rel_constraint}
客体实体类型必须从以下列表中选择。
{obj_constraint}
""",
        "constraint_triple_default": """
**三元组提取约束**：实体类型必须从以下列表中选择。
{constraint}
""",
        "example_wrapper": """
以下是一些示例：
{example}
(示例结束)
""",
        "good_case_wrapper": """
以下是一些示例：
{example}
(示例结束)
请参考示例中的推理步骤和分析，以帮助完成下面的提取任务。
""",
        "bad_case_wrapper": """
以下是一些错误示例：
{example}
(示例结束)
请参考示例中的反思规则和反思步骤，以帮助优化下面的原始结果。
""",
        "case_text": """**文本**：{distilled_text}
{chunk_text}""",
        "case_task": "**任务**：{instruction}",
        "case_original_result": "**原始结果**：{pred}",
        "case_original_answer": "**原始答案**：{pred}",
        "case_correct_answer": "**正确答案**：{truth}",
        "case_analysis": "**分析**：{analysis}",
        "case_reflection": "**反思**：{reflection}",
        "case_index_base": "**任务**：{instruction}",
        "case_index_bad_base": """**任务**：{instruction}

**原始结果**：{pred}""",
        "case_index_constraint": "{constraint}",
        "case_index_bad_constraint": """{constraint}

**原始结果**：{pred}""",
        "good_case_content_base": """{instruction}

{text}

{analysis}

{answer}""",
        "good_case_content_task": """{text}

{constraint}

{analysis}

{answer}""",
        "bad_case_content_base": """{instruction}

{text}

{original_answer}

{reflection}

{correct_answer}""",
        "bad_case_content_task": """{text}

{constraint}

{original_answer}

{reflection}

{correct_answer}"""
    }
}

def _contains_chinese(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def _detect_language(**kwargs):
    from utils.config_manager import ConfigManager

    config_lang = ConfigManager.get_config().get("agent", {}).get("language", "auto")
    if config_lang in {"zh", "en"}:
        return config_lang
    for value in kwargs.values():
        if _contains_chinese(value):
            return "zh"
    return "en"

def get_prompt(name: str, **kwargs) -> str:
    language = _detect_language(**kwargs)
    template = PROMPT_TEMPLATES[language].get(name)
    if template is None:
        raise ValueError(f"Prompt '{name}' not found.")
    return template.format(**kwargs)


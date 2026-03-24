import re
from langchain_core.prompts import PromptTemplate
from .prompt_example import json_schema_examples, code_schema_examples

class BilingualPromptTemplate:
    def __init__(self, input_variables, en_template, cn_template):
        self.en_prompt = PromptTemplate(input_variables=input_variables, template=en_template)
        self.cn_prompt = PromptTemplate(input_variables=input_variables, template=cn_template)

    def is_chinese(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

    def format(self, **kwargs):
        from utils.config_manager import ConfigManager
        
        # Check global language configuration
        config_lang = ConfigManager.get_config().get('agent', {}).get('language', 'auto')
        if config_lang == 'zh':
            return self.cn_prompt.format(**kwargs)
        elif config_lang == 'en':
            return self.en_prompt.format(**kwargs)

        # Determine language based on instruction or text if auto
        text_to_check = kwargs.get("instruction", "")
        if not text_to_check:
            text_to_check = kwargs.get("text", "")
        
        if self.is_chinese(text_to_check):
            return self.cn_prompt.format(**kwargs)
        return self.en_prompt.format(**kwargs)

# ==================================================================== #
#                           SCHEMA AGENT                               #
# ==================================================================== #

# Get Text Analysis
TEXT_ANALYSIS_INSTRUCTION_EN = """
**Instruction**: Please analyze and categorize the given text.
{examples}
**Text**: {text}

**Output Shema**: {schema}
"""

TEXT_ANALYSIS_INSTRUCTION_CN = """
**指令**：请对给定文本进行分析和分类。
{examples}
**文本**：{text}

**输出格式**：{schema}
"""

text_analysis_instruction = BilingualPromptTemplate(
    input_variables=["examples", "text", "schema"],
    en_template=TEXT_ANALYSIS_INSTRUCTION_EN,
    cn_template=TEXT_ANALYSIS_INSTRUCTION_CN
)

# Get Deduced Schema Json
DEDUCE_SCHEMA_JSON_INSTRUCTION_EN = """
**Instruction**: Generate an output format that meets the requirements as described in the task. Pay attention to the following requirements:
    - Format: Return your responses in dictionary format as a JSON object.
    - Content: Do not include any actual data; all attributes values should be set to None.
    - Note: Attributes not mentioned in the task description should be ignored.
{examples}
**Task**: {instruction}

**Text**: {distilled_text}
{text}

Now please deduce the output schema in json format. All attributes values should be set to None.
**Output Schema**:
"""

DEDUCE_SCHEMA_JSON_INSTRUCTION_CN = """
**指令**：根据任务描述生成符合要求的输出格式。请注意以下要求：
    - 格式：以 JSON 对象（字典格式）返回响应。
    - 内容：不要包含任何实际数据；所有属性值应设置为 None。
    - 注意：忽略任务描述中未提及的属性。
{examples}
**任务**：{instruction}

**文本**：{distilled_text}
{text}

现在请以 JSON 格式推导输出结构。所有属性值应设置为 None。
**输出格式**：
"""

deduced_schema_json_instruction = BilingualPromptTemplate(
    input_variables=["examples", "instruction", "distilled_text", "text", "schema"],
    en_template=DEDUCE_SCHEMA_JSON_INSTRUCTION_EN,
    cn_template=DEDUCE_SCHEMA_JSON_INSTRUCTION_CN
)

# Get Deduced Schema Code
DEDUCE_SCHEMA_CODE_INSTRUCTION_EN = """
**Instruction**: Based on the provided text and task description, Define the output schema in Python using Pydantic. Name the final extraction target class as 'ExtractionTarget'.
{examples}
**Task**: {instruction}

**Text**: {distilled_text}
{text}

Now please deduce the output schema. Ensure that the output code snippet is wrapped in '```',and can be directly parsed by the Python interpreter.
**Output Schema**: """

DEDUCE_SCHEMA_CODE_INSTRUCTION_CN = """
**指令**：根据提供的文本和任务描述，使用 Pydantic 定义 Python 输出结构。将最终的提取目标类命名为 'ExtractionTarget'。
{examples}
**任务**：{instruction}

**文本**：{distilled_text}
{text}

现在请推导输出结构。确保输出的代码片段被 '```' 包裹，并且可以直接被 Python 解释器解析。
**输出结构**： """

deduced_schema_code_instruction = BilingualPromptTemplate(
    input_variables=["examples", "instruction", "distilled_text", "text"],
    en_template=DEDUCE_SCHEMA_CODE_INSTRUCTION_EN,
    cn_template=DEDUCE_SCHEMA_CODE_INSTRUCTION_CN
)


# ==================================================================== #
#                         EXTRACTION AGENT                             #
# ==================================================================== #

EXTRACT_INSTRUCTION_EN = """
**Instruction**: You are an agent skilled in information extarction. {instruction}
{examples}
**Text**: {text}
{additional_info}
**Output Schema**: {schema}

Now please extract the corresponding information from the text. Ensure that the information you extract has a clear reference in the given text. Set any property not explicitly mentioned in the text to null.
"""

EXTRACT_INSTRUCTION_CN = """
**指令**：你是一个擅长信息提取的智能体。{instruction}
{examples}
**文本**：{text}
{additional_info}
**输出格式**：{schema}

现在请从文本中提取相应的信息。确保你提取的信息在给定文本中有明确的引用。将文本中未明确提及的任何属性设置为 null。
"""

extract_instruction = BilingualPromptTemplate(
    input_variables=["instruction", "examples", "text", "schema", "additional_info"],
    en_template=EXTRACT_INSTRUCTION_EN,
    cn_template=EXTRACT_INSTRUCTION_CN
)


# ==================================================================== #
#                          REFLECION AGENT                             #
# ==================================================================== #
REFLECT_INSTRUCTION_EN = """**Instruction**: You are an agent skilled in reflection and optimization based on the original result. Refer to **Reflection Reference** to identify potential issues in the current extraction results.

**Reflection Reference**: {examples}

Now please review each element in the extraction result. Identify and improve any potential issues in the result based on the reflection. NOTE: If the original result is correct, no modifications are needed!

**Task**: {instruction}

**Text**: {text}

**Output Schema**: {schema}

**Original Result**: {result}

"""

REFLECT_INSTRUCTION_CN = """**指令**：你是一个擅长基于原始结果进行反思和优化的智能体。请参考**反思参考**来识别当前提取结果中的潜在问题。

**反思参考**：{examples}

现在请审查提取结果中的每个元素。根据反思识别并改进结果中的任何潜在问题。注意：如果原始结果是正确的，则不需要进行任何修改！

**任务**：{instruction}

**文本**：{text}

**输出格式**：{schema}

**原始结果**：{result}

"""

reflect_instruction = BilingualPromptTemplate(
    input_variables=["instruction", "examples", "text", "schema", "result"],
    en_template=REFLECT_INSTRUCTION_EN,
    cn_template=REFLECT_INSTRUCTION_CN
)

SUMMARIZE_INSTRUCTION_EN = """
**Instruction**: Below is a list of results obtained after segmenting and extracting information from a long article. Please consolidate all the answers to generate a final response.

**Task**: {instruction}

**Result List**: {answer_list}
{additional_info}
**Output Schema**: {schema}
Now summarize the information from the Result List.
"""

SUMMARIZE_INSTRUCTION_CN = """
**指令**：以下是从长篇文章中分段提取信息后获得的结果列表。请整合所有答案以生成最终的响应。

**任务**：{instruction}

**结果列表**：{answer_list}
{additional_info}
**输出格式**：{schema}
现在请总结结果列表中的信息。
"""

summarize_instruction = BilingualPromptTemplate(
    input_variables=["instruction", "answer_list", "additional_info", "schema"],
    en_template=SUMMARIZE_INSTRUCTION_EN,
    cn_template=SUMMARIZE_INSTRUCTION_CN
)


# ==================================================================== #
#                            CASE REPOSITORY                           #
# ==================================================================== #

GOOD_CASE_ANALYSIS_INSTRUCTION_EN = """
**Instruction**: Below is an information extraction task and its corresponding correct answer. Provide the reasoning steps that led to the correct answer, along with brief explanation of the answer. Your response should be brief and organized.

**Task**: {instruction}

**Text**: {text}
{additional_info}
**Correct Answer**: {result}

Now please generate the reasoning steps and breif analysis of the **Correct Answer** given above. DO NOT generate your own extraction result.
**Analysis**:
"""

GOOD_CASE_ANALYSIS_INSTRUCTION_CN = """
**指令**：以下是一个信息提取任务及其对应的正确答案。请提供得出正确答案的推理步骤，并对答案进行简要解释。你的回答应简明扼要且有条理。

**任务**：{instruction}

**文本**：{text}
{additional_info}
**正确答案**：{result}

现在请生成对上述**正确答案**的推理步骤和简要分析。不要生成你自己的提取结果。
**分析**：
"""

good_case_analysis_instruction = BilingualPromptTemplate(
    input_variables=["instruction", "text", "result", "additional_info"],
    en_template=GOOD_CASE_ANALYSIS_INSTRUCTION_EN,
    cn_template=GOOD_CASE_ANALYSIS_INSTRUCTION_CN
)

BAD_CASE_REFLECTION_INSTRUCTION_EN = """
**Instruction**: Based on the task description, compare the original answer with the correct one. Your output should be a brief reflection or concise summarized rules.

**Task**: {instruction}

**Text**: {text}
{additional_info}
**Original Answer**: {original_answer}

**Correct Answer**: {correct_answer}

Now please generate a brief and organized reflection. DO NOT generate your own extraction result.
**Reflection**:
"""

BAD_CASE_REFLECTION_INSTRUCTION_CN = """
**指令**：根据任务描述，将原始答案与正确答案进行比较。你的输出应该是一个简短的反思或简明扼要的总结规则。

**任务**：{instruction}

**文本**：{text}
{additional_info}
**原始答案**：{original_answer}

**正确答案**：{correct_answer}

现在请生成一个简明有条理的反思。不要生成你自己的提取结果。
**反思**：
"""

bad_case_reflection_instruction = BilingualPromptTemplate(
    input_variables=["instruction", "text", "original_answer", "correct_answer", "additional_info"],
    en_template=BAD_CASE_REFLECTION_INSTRUCTION_EN,
    cn_template=BAD_CASE_REFLECTION_INSTRUCTION_CN
)

# ==================================================================== #
#                          DEFAULT PROMPTS                             #
# ==================================================================== #

default_schema = BilingualPromptTemplate(
    input_variables=[],
    en_template="The final extraction result should be formatted as a JSON object.",
    cn_template="最终的提取结果应格式化为 JSON 对象。"
)

default_ner = BilingualPromptTemplate(
    input_variables=[],
    en_template="Extract the Named Entities in the given text.",
    cn_template="提取给定文本中的命名实体。"
)

default_re = BilingualPromptTemplate(
    input_variables=[],
    en_template="Extract Relationships between Named Entities in the given text.",
    cn_template="提取给定文本中命名实体之间的关系。"
)

default_ee = BilingualPromptTemplate(
    input_variables=[],
    en_template="Extract the Events in the given text.",
    cn_template="提取给定文本中的事件。"
)

default_triple = BilingualPromptTemplate(
    input_variables=[],
    en_template="Extract the Triples (subject, relation, object) from the given text, hope that all the relationships for each entity can be extracted.",
    cn_template="从给定文本中提取三元组（主语，关系，宾语），希望能够提取出每个实体的所有关系。"
)

PROMPT_REGISTRY = {
    "text_analysis": text_analysis_instruction,
    "deduced_schema_json": deduced_schema_json_instruction,
    "deduced_schema_code": deduced_schema_code_instruction,
    "extract": extract_instruction,
    "reflect": reflect_instruction,
    "summarize": summarize_instruction,
    "good_case_analysis": good_case_analysis_instruction,
    "bad_case_reflection": bad_case_reflection_instruction,
    "default_schema": default_schema,
    "default_ner": default_ner,
    "default_re": default_re,
    "default_ee": default_ee,
    "default_triple": default_triple
}



import re
from models.llm_def import BaseEngine
from models.prompt_example import json_schema_examples, code_schema_examples
from models.prompt_template import PROMPT_REGISTRY
from utils.data_def import DataPoint
from utils.process import extract_json_dict, example_wrapper, current_function_name, chunk_file, chunk_str
from utils.config_manager import ConfigManager
from .knowledge_base import schema_repository
from .task_strategy import TaskStrategyFactory
from langchain_core.output_parsers import JsonOutputParser
from .base_agent import BaseAgent

class SchemaAgent(BaseAgent):
    def __init__(self, llm: BaseEngine):
        super().__init__(llm)
        self.schema_repo = schema_repository

    def serialize_schema(self, schema) -> str:
        if isinstance(schema, (str, list, dict, set, tuple)):
            return schema
        try:
            parser = JsonOutputParser(pydantic_object = schema)
            schema_description = parser.get_format_instructions()
            schema_content = re.findall(r'```(.*?)```', schema_description, re.DOTALL)
            explanation = "For example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}}, the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance."
            schema = f"{schema_content}\n\n{explanation}"
        except:
            return schema
        return schema

    def redefine_text(self, text_analysis):
        try:
            field = text_analysis['field']
            genre = text_analysis['genre']
        except:
            return text_analysis
        prompt = f"This text is from the field of {field} and represents the genre of {genre}."
        return prompt

    def __preprocess_text(self, data: DataPoint):
        if data.use_file:
            data.chunk_text_list = chunk_file(data.file_path)
        else:
            data.chunk_text_list = chunk_str(data.text)
            
        strategy = TaskStrategyFactory.get_strategy(data.task)
        print_schema = strategy.get_print_schema()
        if print_schema:
            data.print_schema = print_schema
            
        return data

    def get_default_schema(self, data: DataPoint):
        data = self.__preprocess_text(data)
        default_schema = PROMPT_REGISTRY['default_schema'].format()
        data.set_schema(default_schema)
        function_name = current_function_name()
        data.update_trajectory(function_name, default_schema)
        return data

    def get_retrieved_schema(self, data: DataPoint):
        self.__preprocess_text(data)
        schema_name = data.output_schema
        schema_class = getattr(self.schema_repo, schema_name, None)
        if schema_class is not None:
            schema = self.serialize_schema(schema_class)
            default_schema = PROMPT_REGISTRY['default_schema'].format()
            data.set_schema(f"{default_schema}\n{schema}")
            function_name = current_function_name()
            data.update_trajectory(function_name, schema)
        else:
            return self.get_default_schema(data)
        return data

    def get_deduced_schema(self, data: DataPoint):
        self.__preprocess_text(data)
        target_text = data.chunk_text_list[0]
        
        # 1. get text analysis
        output_schema = self.serialize_schema(self.schema_repo.TextDescription)
        analysed_text_dict = self.invoke_llm(
            mode="text_analysis",
            examples="",
            text=target_text,
            schema=output_schema
        )
        analysed_text = self.redefine_text(analysed_text_dict)
        
        if len(data.chunk_text_list) > 1:
            prefix = "Below is a portion of the text to be extracted. "
            analysed_text = f"{prefix}\n{target_text}"
        distilled_text = analysed_text

        # 2. get deduced schema code
        response = self.invoke_llm(
            mode="deduced_schema_code",
            extract_json=False,
            examples=example_wrapper(code_schema_examples),
            instruction=data.instruction,
            distilled_text=distilled_text,
            text=target_text
        )
        
        code = ""
        deduced_schema = None
        code_blocks = re.findall(r'```[^\n]*\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            try:
                code_block = code_blocks[-1]
                namespace = {}
                exec(code_block, namespace)
                schema_cls = namespace.get('ExtractionTarget')
                if schema_cls is not None:
                    index = code_block.find("class")
                    code = code_block[index:]
                    print(f"Deduced Schema in Code: \n{code}\n\n")
                    deduced_schema = self.serialize_schema(schema_cls)
            except Exception as e:
                print(e)
                
        if deduced_schema is None:
            response = self.invoke_llm(
                mode="deduced_schema_json",
                examples=example_wrapper(json_schema_examples),
                instruction=data.instruction,
                distilled_text=distilled_text,
                text=target_text
            )
            code = response
            deduced_schema = response
            print(f"Deduced Schema in Json: \n{response}\n\n")

        data.print_schema = code
        data.set_distilled_text(distilled_text)
        default_schema = PROMPT_REGISTRY['default_schema'].format()
        data.set_schema(f"{default_schema}\n{deduced_schema}")
        data.update_trajectory(current_function_name(), deduced_schema)
        return data

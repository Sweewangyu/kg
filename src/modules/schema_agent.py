import re
from models.llm_def import BaseEngine
from models.prompt_example import json_schema_examples, code_schema_examples
from models.prompt_template import get_prompt
from utils.data_def import DataPoint
from utils.process import example_wrapper, current_function_name, chunk_file, chunk_str
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
            schema = get_prompt("schema_serialized", schema_content=schema_content)
        except:
            return schema
        return schema

    def redefine_text(self, text_analysis):
        try:
            field = text_analysis['field']
            genre = text_analysis['genre']
        except:
            return text_analysis
        return get_prompt("text_analysis_summary", field=field, genre=genre)

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
        default_schema = get_prompt("default_schema")
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
            data.set_schema(get_prompt("schema_with_default", default_schema=get_prompt("default_schema"), schema=schema))
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
            analysed_text = get_prompt("partial_text", text=target_text)
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
        data.set_schema(get_prompt("schema_with_default", default_schema=get_prompt("default_schema"), schema=deduced_schema))
        data.update_trajectory(current_function_name(), deduced_schema)
        return data

from models.llm_def import BaseEngine
from utils.data_def import DataPoint
from utils.process import extract_json_dict, current_function_name
from .knowledge_base.case_repository import CaseRepositoryHandler
from .task_strategy import TaskStrategyFactory
from .base_agent import BaseAgent

class ExtractionAgent(BaseAgent):
    def __init__(self, llm: BaseEngine, case_repo: CaseRepositoryHandler):
        super().__init__(llm)
        self.case_repo = case_repo

    def __get_constraint(self, data: DataPoint):
        if data.constraint == "":
            return data
            
        strategy = TaskStrategyFactory.get_strategy(data.task)
        data.constraint = strategy.format_constraint(data.constraint)
        return data

    def extract_information_direct(self, data: DataPoint):
        data = self.__get_constraint(data)
        result_list = []
        for chunk_text in data.chunk_text_list:
            extract_direct_result = self.invoke_llm(
                mode="extract",
                instruction=data.instruction, 
                text=chunk_text, 
                schema=data.output_schema, 
                examples="", 
                constraint=data.constraint,
                task=data.task
            )
            result_list.append(extract_direct_result)
        function_name = current_function_name()
        data.set_result_list(result_list)
        data.update_trajectory(function_name, result_list)
        return data

    def extract_information_with_case(self, data: DataPoint):
        data = self.__get_constraint(data)
        result_list = []
        for chunk_text in data.chunk_text_list:
            examples = self.case_repo.query_good_case(data)
            examples_str = "\n\n".join(examples) if examples else ""
            extract_case_result = self.invoke_llm(
                mode="extract",
                instruction=data.instruction, 
                text=chunk_text, 
                schema=data.output_schema, 
                examples=examples_str, 
                constraint=data.constraint,
                task=data.task
            )
            result_list.append(extract_case_result)
        function_name = current_function_name()
        data.set_result_list(result_list)
        data.update_trajectory(function_name, result_list)
        return data

    def summarize_answer(self, data: DataPoint):
        if len(data.result_list) == 0:
            return data
        if len(data.result_list) == 1:
            data.set_pred(data.result_list[0])
            return data
        summarized_result = self.invoke_llm(
            mode="summarize",
            instruction=data.instruction, 
            answer_list=data.result_list, 
            schema=data.output_schema, 
            constraint=data.constraint,
            task=data.task
        )
        funtion_name = current_function_name()
        data.set_pred(summarized_result)
        data.update_trajectory(funtion_name, summarized_result)
        return data

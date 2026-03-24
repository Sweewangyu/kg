import json
from abc import ABC, abstractmethod
from models import BaseEngine
from utils.data_def import DataPoint, TaskType
from utils.process import extract_json_dict
from utils.config_manager import ConfigManager
from modules.schema_agent import SchemaAgent
from modules.extraction_agent import ExtractionAgent
from modules.reflection_agent import ReflectionAgent
from modules.knowledge_base.case_repository import CaseRepositoryHandler
from modules.task_strategy import TaskStrategyFactory
from construct.convert import generate_cypher_statements, execute_cypher_statements


class PipelineStep(ABC):
    @abstractmethod
    def execute(self, data: DataPoint) -> DataPoint:
        pass

class AgentStep(PipelineStep):
    def __init__(self, agent, method_name: str):
        self.agent = agent
        self.method_name = method_name
        
    def execute(self, data: DataPoint) -> DataPoint:
        method = getattr(self.agent, self.method_name, None)
        if not method:
            raise AttributeError(f"Method '{self.method_name}' not found in agent.")
        return method(data)


class Pipeline:
    def __init__(self, llm: BaseEngine):
        self.llm = llm
        self.case_repo = CaseRepositoryHandler(llm = llm)
        self.schema_agent = SchemaAgent(llm = llm)
        self.extraction_agent = ExtractionAgent(llm = llm, case_repo = self.case_repo)
        self.reflection_agent = ReflectionAgent(llm = llm, case_repo = self.case_repo)

    def __init_method(self, data: DataPoint, process_method):
        default_order = ["schema_agent", "extraction_agent", "reflection_agent"]
        if "schema_agent" not in process_method:
            process_method["schema_agent"] = "get_default_schema"
        if data.task != "Base":
            process_method["schema_agent"] = "get_retrieved_schema"
        if "extraction_agent" not in process_method:
            process_method["extraction_agent"] = "extract_information_direct"
        sorted_process_method = {key: process_method[key] for key in default_order if key in process_method}
        return sorted_process_method

    def __init_data(self, data: DataPoint):
        strategy = TaskStrategyFactory.get_strategy(data.task)
        instruction = strategy.get_instruction()
        if instruction:
            data.instruction = instruction
        output_schema = strategy.get_output_schema()
        if output_schema:
            data.output_schema = output_schema
        return data

    def __build_data(self, task: TaskType, instruction: str, text: str, output_schema: str, constraint: str, use_file: bool, file_path: str, truth: str):
        data = DataPoint(
            task=task,
            instruction=instruction,
            text=text,
            output_schema=output_schema,
            constraint=constraint,
            use_file=use_file,
            file_path=file_path,
            truth=truth
        )
        return self.__init_data(data)

    def __resolve_process_method(self, data: DataPoint, mode):
        config = ConfigManager.get_config()
        if mode in config['agent']['mode']:
            process_method = config['agent']['mode'][mode].copy()
        else:
            process_method = mode
        return self.__init_method(data, process_method)

    def __build_steps(self, sorted_process_method):
        steps = []
        for agent_name, method_name in sorted_process_method.items():
            agent = getattr(self, agent_name, None)
            if not agent:
                raise AttributeError(f"{agent_name} does not exist.")
            steps.append(AgentStep(agent, method_name))
        return steps

    def __run_steps(self, data: DataPoint, steps):
        has_printed_schema = False
        for step in steps:
            data = step.execute(data)
            if not has_printed_schema and data.print_schema:
                print("Schema: \n", data.print_schema)
                has_printed_schema = True
        return self.extraction_agent.summarize_answer(data)

    def __show_result(self, data: DataPoint, show_trajectory: bool):
        if show_trajectory:
            print("Extraction Trajectory: \n", json.dumps(data.get_result_trajectory(), indent=2))
        extraction_result = json.dumps(data.pred, indent=2)
        print("Extraction Result: \n", extraction_result)
        return extraction_result

    def __construct_kg(self, extraction_result, construct):
        myurl = construct['url']
        myusername = construct['username']
        mypassword = construct['password']
        print(f"Construct KG in your {construct['database']} now...")
        cypher_statements = generate_cypher_statements(extraction_result)
        execute_cypher_statements(uri=myurl, user=myusername, password=mypassword, cypher_statements=cypher_statements)

    def __update_case(self, data: DataPoint):
        if data.truth == "":
            truth = input("Please enter the correct answer you prefer, or just press Enter to accept the current answer: ")
            if truth.strip() == "":
                data.truth = data.pred
            else:
                data.truth = extract_json_dict(truth)
        self.case_repo.update_case(data)

    # main entry
    def get_extract_result(self,
                           task: TaskType,
                           three_agents = None,
                           construct = None,
                           instruction: str = "",
                           text: str = "",
                           output_schema: str = "",
                           constraint: str = "",
                           use_file: bool = False,
                           file_path: str = "",
                           truth: str = "",
                           mode: str = "quick",
                           update_case: bool = False,
                           show_trajectory: bool = False,
                           iskg: bool = False,
                           ):
        construct = construct or {}
        data = self.__build_data(task, instruction, text, output_schema, constraint, use_file, file_path, truth)
        sorted_process_method = self.__resolve_process_method(data, mode)
        print("Process Method: ", sorted_process_method)
        steps = self.__build_steps(sorted_process_method)
        data = self.__run_steps(data, steps)
        extraction_result = self.__show_result(data, show_trajectory)
        if iskg:
            self.__construct_kg(extraction_result, construct)
        if update_case:
            self.__update_case(data)
        result = data.pred
        trajectory = data.get_result_trajectory()
        return result, trajectory

import json
from abc import ABC, abstractmethod
from models.prompt_template import get_prompt

class TaskStrategy(ABC):
    @abstractmethod
    def get_print_schema(self) -> str:
        """Returns the print schema string for the specific task."""
        pass

    @abstractmethod
    def format_constraint(self, raw_constraint) -> str:
        """Formats the raw constraint into a task-specific constraint string."""
        pass

    @abstractmethod
    def get_instruction(self) -> str:
        pass

    @abstractmethod
    def get_output_schema(self) -> str:
        pass

class NERTaskStrategy(TaskStrategy):
    def get_print_schema(self) -> str:
        return """
class Entity(BaseModel):
    name : str = Field(description="The specific name of the entity. ")
    type : str = Field(description="The type or category that the entity belongs to.")
class EntityList(BaseModel):
    entity_list : List[Entity] = Field(description="Named entities appearing in the text.")
            """

    def format_constraint(self, raw_constraint) -> str:
        constraint = json.dumps(raw_constraint)
        if "**Entity Type Constraint**" in constraint or "**实体类型约束**" in constraint:
            return raw_constraint
        return get_prompt("constraint_ner", constraint=constraint)

    def get_instruction(self) -> str:
        return get_prompt("default_ner")

    def get_output_schema(self) -> str:
        return "EntityList"

class RETaskStrategy(TaskStrategy):
    def get_print_schema(self) -> str:
        return """
class Relation(BaseModel):
    head : str = Field(description="The starting entity in the relationship.")
    tail : str = Field(description="The ending entity in the relationship.")
    relation : str = Field(description="The predicate that defines the relationship between the two entities.")

class RelationList(BaseModel):
    relation_list : List[Relation] = Field(description="The collection of relationships between various entities.")
            """

    def format_constraint(self, raw_constraint) -> str:
        constraint = json.dumps(raw_constraint)
        if "**Relation Type Constraint**" in constraint or "**关系类型约束**" in constraint:
            return raw_constraint
        return get_prompt("constraint_re", constraint=constraint)

    def get_instruction(self) -> str:
        return get_prompt("default_re")

    def get_output_schema(self) -> str:
        return "RelationList"

class EETaskStrategy(TaskStrategy):
    def get_print_schema(self) -> str:
        return """
class Event(BaseModel):
    event_type : str = Field(description="The type of the event.")
    event_trigger : str = Field(description="A specific word or phrase that indicates the occurrence of the event.")
    event_argument : dict = Field(description="The arguments or participants involved in the event.")

class EventList(BaseModel):
    event_list : List[Event] = Field(description="The events presented in the text.")
            """

    def format_constraint(self, raw_constraint) -> str:
        constraint = json.dumps(raw_constraint)
        if "**Event Extraction Constraint**" in constraint or "**事件提取约束**" in constraint:
            return raw_constraint
        return get_prompt("constraint_ee", constraint=constraint)

    def get_instruction(self) -> str:
        return get_prompt("default_ee")

    def get_output_schema(self) -> str:
        return "EventList"

class TripleTaskStrategy(TaskStrategy):
    def get_print_schema(self) -> str:
        return """
class Triple(BaseModel):
    head: str = Field(description="The subject or head of the triple.")
    head_type: str = Field(description="The type of the subject entity.")
    relation: str = Field(description="The predicate or relation between the entities.")
    relation_type: str = Field(description="The type of the relation.")
    tail: str = Field(description="The object or tail of the triple.")
    tail_type: str = Field(description="The type of the object entity.")
class TripleList(BaseModel):
    triple_list: List[Triple] = Field(description="The collection of triples and their types presented in the text.")
"""

    def format_constraint(self, raw_constraint) -> str:
        constraint_str = json.dumps(raw_constraint)
        if "**Triple Extraction Constraint**" in constraint_str or "**三元组提取约束**" in constraint_str:
            return raw_constraint
        
        if isinstance(raw_constraint, list):
            if len(raw_constraint) == 1:
                return get_prompt("constraint_triple_entity", constraint=constraint_str)
            elif len(raw_constraint) == 2:
                if not raw_constraint[0]:
                    return get_prompt("constraint_triple_relation", constraint=json.dumps(raw_constraint[1]))
                elif not raw_constraint[1]:
                    return get_prompt("constraint_triple_entity", constraint=json.dumps(raw_constraint[0]))
                else:
                    return get_prompt("constraint_triple_ent_rel", ent_constraint=json.dumps(raw_constraint[0]), rel_constraint=json.dumps(raw_constraint[1]))
            elif len(raw_constraint) == 3:
                if not raw_constraint[0]:
                    return get_prompt("constraint_triple_rel_obj", rel_constraint=json.dumps(raw_constraint[1]), obj_constraint=json.dumps(raw_constraint[2]))
                elif not raw_constraint[1]:
                    return get_prompt("constraint_triple_subj_obj", subj_constraint=json.dumps(raw_constraint[0]), obj_constraint=json.dumps(raw_constraint[2]))
                elif not raw_constraint[2]:
                    return get_prompt("constraint_triple_subj_rel", subj_constraint=json.dumps(raw_constraint[0]), rel_constraint=json.dumps(raw_constraint[1]))
                else:
                    return get_prompt("constraint_triple_all", subj_constraint=json.dumps(raw_constraint[0]), rel_constraint=json.dumps(raw_constraint[1]), obj_constraint=json.dumps(raw_constraint[2]))
        
        return get_prompt("constraint_triple_default", constraint=constraint_str)

    def get_instruction(self) -> str:
        return get_prompt("default_triple")

    def get_output_schema(self) -> str:
        return "TripleList"

class BaseTaskStrategy(TaskStrategy):
    def get_print_schema(self) -> str:
        return ""
        
    def format_constraint(self, raw_constraint) -> str:
        return raw_constraint

    def get_instruction(self) -> str:
        return ""

    def get_output_schema(self) -> str:
        return ""

class TaskStrategyFactory:
    @staticmethod
    def get_strategy(task_name: str) -> TaskStrategy:
        strategies = {
            "NER": NERTaskStrategy,
            "RE": RETaskStrategy,
            "EE": EETaskStrategy,
            "Triple": TripleTaskStrategy,
            "Base": BaseTaskStrategy
        }
        return strategies.get(task_name, BaseTaskStrategy)()

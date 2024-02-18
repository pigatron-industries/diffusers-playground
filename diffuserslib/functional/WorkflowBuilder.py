from diffuserslib.functional.FunctionalNode import FunctionalNode
from typing import Any


class WorkflowBuilder:
    def __init__(self, name:str="Workflow Name", type:type=Any, workflow:bool=True, subworkflow:bool=False):
        self.name = name
        self.type = type
        self.workflow = workflow
        self.subworkflow = subworkflow

    def build(self) -> FunctionalNode:
        raise Exception("Not implemented")
from .FunctionalNode import FunctionalNode, ParameterInfos, TypeInfo
from typing import Any
from dataclasses import dataclass
from PIL import Image



@dataclass
class WorkflowRunData:
    params:ParameterInfos
    output: Any|None = None
    save_file:str|None = None


class WorkflowRunner:
    def __init__(self):
        self.rundata = []

    def setWorkflow(self, workflow:FunctionalNode):
        self.workflow = workflow

    def clearRunData(self):
        self.rundata = []

    def run(self, workflow:FunctionalNode, batch_size:int = 1):
        for i in range(batch_size):
            rundata = WorkflowRunData(workflow.getStaticParams())
            self.rundata.append(rundata)
            rundata.output = workflow()
        
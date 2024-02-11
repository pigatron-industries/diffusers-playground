from .FunctionalNode import FunctionalNode, ParameterInfos, TypeInfo
from typing import Any
from dataclasses import dataclass

from time import sleep


@dataclass
class WorkflowRunData:
    params:ParameterInfos
    output: Any|None = None
    save_file:str|None = None


class WorkflowRunner:
    def __init__(self):
        self.rundata = []
        self.running = False

    def setWorkflow(self, workflow:FunctionalNode):
        self.workflow = workflow

    def clearRunData(self):
        self.rundata = []

    def run(self, workflow:FunctionalNode, batch_size:int = 1):
        self.running = True
        print(f"Running workflow {workflow.node_name} with batch size {batch_size}")
        for i in range(batch_size):
            print(f"Running workflow {workflow.node_name} batch {i+1} of {batch_size}")
            rundata = WorkflowRunData(workflow.getStaticParams())
            self.rundata.append(rundata)
            rundata.output = workflow()
            if(self.running == False):
                break
        self.running = False
        
    def stop(self):
        self.running = False
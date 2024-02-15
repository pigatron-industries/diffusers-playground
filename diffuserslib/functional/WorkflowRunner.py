from .FunctionalNode import FunctionalNode, ParameterInfos
from typing import Any, Dict, Self
from dataclasses import dataclass
from PIL import Image
import time

@dataclass
class WorkflowRunData:
    params:ParameterInfos
    timestamp:int
    output: Any|None = None
    save_file:str|None = None


class WorkflowRunner:
    workflowrunner:Self|None = None

    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.rundata:Dict[int, WorkflowRunData] = {}
        self.running = False

    def setWorkflow(self, workflow:FunctionalNode):
        self.workflow = workflow

    def clearRunData(self):
        self.rundata = {}

    def run(self, workflow:FunctionalNode, batch_size:int = 1):
        self.running = True
        print(f"Running workflow {workflow.node_name} with batch size {batch_size}")
        for i in range(batch_size):
            print(f"Running workflow {workflow.node_name} batch {i+1} of {batch_size}")
            rundata = WorkflowRunData(workflow.getStaticParams(), int(time.time_ns()/1000))
            self.rundata[rundata.timestamp] = rundata
            rundata.output = workflow()
            if(self.running == False):
                break
        self.running = False
        
    def stop(self):
        self.running = False

    def save(self, timestamp:int):
        save_file = f"{self.output_dir}/output_{timestamp}"
        output = self.rundata[timestamp].output
        if(output is not None):
            if(isinstance(output, Image.Image)):
                output.save(f"{save_file}.png")
                self.rundata[timestamp].save_file = f"{save_file}.png"
                # TODO: save params to file
                print(f"Saved output to {save_file}")
            else:
                raise Exception("Output is not an image")

from .FunctionalNode import FunctionalNode, ParameterInfos
from typing import Any, Dict, Self
from dataclasses import dataclass
from PIL import Image
import time
import yaml
import numpy as np

@dataclass
class WorkflowRunData:
    timestamp:int
    params:Dict[str,Dict[str,Any]]|None = None
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
            rundata = WorkflowRunData(int(time.time_ns()/1000))
            self.rundata[rundata.timestamp] = rundata
            rundata.output = workflow()
            rundata.params = workflow.getEvaluatedParamValuesRecursive()
            if(self.running == False):
                break
        self.running = False
        
    def stop(self):
        self.running = False

    def save(self, timestamp:int):
        save_file = f"{self.output_dir}/output_{timestamp}"
        rundata = self.rundata[timestamp]
        if(rundata.output is not None):
            if(isinstance(rundata.output, Image.Image)):
                rundata.output.save(f"{save_file}.png")
                rundata.save_file = f"{save_file}.png"
                file = open(f"{save_file}.yaml", "w")
                yaml.dump(rundata.params, file)
                print(rundata.params)
                print(f"Saved output to {save_file}")
            else:
                raise Exception("Output is not an image")

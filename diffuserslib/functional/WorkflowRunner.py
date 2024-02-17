from .FunctionalNode import FunctionalNode
from typing import Any, Dict, Self, List
from dataclasses import dataclass
from PIL import Image
import time
import yaml
import numpy as np
import copy
import sys


@dataclass
class WorkflowRunData:
    timestamp:int
    params:Dict[str,Dict[str,Any]]|None = None
    output: Any|None = None
    save_file:str|None = None
    error:Exception|None = None


@dataclass
class WorkflowQueueData:
    workflow:FunctionalNode
    batch_size:int


@dataclass
class ProgressData:
    jobs_remaining:int
    jobs_completed:int



class WorkflowRunner:
    workflowrunner:Self|None = None

    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.rundata:Dict[int, WorkflowRunData] = {}
        self.queue:List[WorkflowQueueData] = []
        self.progress:ProgressData = ProgressData(0,0)
        self.stopping = False
        self.running = False

    def setWorkflow(self, workflow:FunctionalNode):
        self.workflow = workflow

    def clearRunData(self):
        self.rundata = {}

    def run(self, workflow:FunctionalNode, batch_size:int = 1):
        self.queue.append(WorkflowQueueData(copy.deepcopy(workflow), batch_size))
        self.progress.jobs_remaining += batch_size
        if(self.running):
            return

        self.running = True
        print(f"Running workflow {workflow.node_name} with batch size {batch_size}")
        while len(self.queue) > 0:
            queue_data = self.queue.pop(0)
            for i in range(batch_size):
                print(f"Running workflow {queue_data.workflow.node_name} batch {i+1} of {queue_data.batch_size}")
                rundata = WorkflowRunData(int(time.time_ns()/1000))
                self.rundata[rundata.timestamp] = rundata
                try:
                    rundata.output = queue_data.workflow()
                    rundata.params = queue_data.workflow.getEvaluatedParamValues()
                    self.progress.jobs_completed += 1
                    self.progress.jobs_remaining -= 1
                except Exception as e:
                    rundata.error = e
                    print(f"Error running workflow {queue_data.workflow.node_name}: {e}")
                    self.stopping = True
                if(self.stopping == True):
                    break
        self.running = False
        self.stopping = False
        self.progress = ProgressData(0,0)
        
    def stop(self):
        self.stopping = True

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

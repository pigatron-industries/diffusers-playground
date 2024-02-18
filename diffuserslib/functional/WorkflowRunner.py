from .FunctionalNode import FunctionalNode
from typing import Any, Dict, Self, List
from dataclasses import dataclass, field
from PIL import Image
import time
import yaml
import copy
import traceback


@dataclass
class WorkflowRunData:
    timestamp:int
    params:Dict[str,Dict[str,Any]]|None = None
    output: Any|None = None
    save_file:str|None = None
    error:Exception|None = None


@dataclass
class WorkflowBatchData:
    id:int
    workflow:FunctionalNode
    batch_size:int
    rundata:Dict[int, WorkflowRunData] = field(default_factory= lambda: {})
    error:Exception|None = None


@dataclass
class ProgressData:
    jobs_remaining:int
    jobs_completed:int



class WorkflowRunner:
    workflowrunner:Self|None = None

    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.rundata:Dict[int, WorkflowRunData] = {}
        self.batchrundata:Dict[int, WorkflowBatchData] = {}
        self.batchqueue:List[WorkflowBatchData] = []
        self.progress:ProgressData = ProgressData(0,0)
        self.stopping = False
        self.running = False
        self.batchcount = 0

    def setWorkflow(self, workflow:FunctionalNode):
        self.workflow = workflow

    def clearRunData(self):
        self.rundata = {}
        self.batchrundata = {}

    def run(self, workflow:FunctionalNode, batch_size:int = 1):
        self.batchqueue.append(WorkflowBatchData(self.batchcount, copy.deepcopy(workflow), batch_size))
        self.batchcount += 1
        self.progress.jobs_remaining += batch_size
        if(self.running):
            return

        self.running = True
        while len(self.batchqueue) > 0:
            current_batch = self.batchqueue.pop(0)
            self.batchrundata[current_batch.id] = current_batch
            print(f"Running workflow {workflow.node_name} with batch size {current_batch.batch_size}")
            for i in range(current_batch.batch_size):
                print(f"Running workflow {current_batch.workflow.node_name} batch {i+1} of {current_batch.batch_size}")
                rundata = WorkflowRunData(int(time.time_ns()/1000))
                current_batch.rundata[rundata.timestamp] = rundata
                self.rundata[rundata.timestamp] = rundata
                try:
                    rundata.output = current_batch.workflow()
                except Exception as e:
                    rundata.error = e
                    print(f"Error running workflow {current_batch.workflow.node_name}: {e}")
                    traceback.print_exc()
                    break
                rundata.params = current_batch.workflow.getEvaluatedParamValues()
                self.progress.jobs_completed = len(self.rundata)
                self.progress.jobs_remaining = sum([batch.batch_size for batch in self.batchqueue]) + current_batch.batch_size - len(current_batch.rundata)
                if(self.stopping == True):
                    break
        self.running = False
        self.stopping = False

        
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

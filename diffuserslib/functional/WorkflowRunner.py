from diffuserslib.functional import Video, FunctionalNode, WorkflowProgress
from typing import Any, Dict, Self, List
from dataclasses import dataclass, field
from PIL import Image
import time
import yaml
import copy
import traceback
import shutil


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
    rundata:Dict[int, WorkflowRunData] = field(default_factory = lambda: {})
    error:Exception|None = None


@dataclass
class BatchProgressData:
    jobs_remaining:int
    jobs_completed:int
    run_progress:WorkflowProgress|None = None



class WorkflowRunner:
    workflowrunner:Self|None = None

    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        self.rundata:Dict[int, WorkflowRunData] = {}
        self.batchrundata:Dict[int, WorkflowBatchData] = {}
        self.batchqueue:List[WorkflowBatchData] = []
        self.batchcurrent:WorkflowBatchData|None = None
        self.progress:BatchProgressData = BatchProgressData(0,0)
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
            self.batchcurrent = self.batchqueue.pop(0)
            self.batchrundata[self.batchcurrent.id] = self.batchcurrent
            print(f"Running workflow {workflow.node_name} with batch size {self.batchcurrent.batch_size}")
            for i in range(self.batchcurrent.batch_size):
                print(f"Running workflow {self.batchcurrent.workflow.node_name} batch {i+1} of {self.batchcurrent.batch_size}")
                rundata = WorkflowRunData(int(time.time_ns()/1000))
                self.batchcurrent.rundata[rundata.timestamp] = rundata
                self.rundata[rundata.timestamp] = rundata
                try:
                    self.batchcurrent.workflow.reset()
                    rundata.output = self.batchcurrent.workflow()
                except Exception as e:
                    rundata.error = e
                    print(f"Error running workflow {self.batchcurrent.workflow.node_name}: {e}")
                    traceback.print_exc()
                    break
                rundata.params = self.batchcurrent.workflow.getEvaluatedParamValues()
                self.progress.jobs_completed = len(self.rundata)
                self.progress.jobs_remaining = sum([batch.batch_size for batch in self.batchqueue]) + self.batchcurrent.batch_size - len(self.batchcurrent.rundata)
                if(self.stopping == True):
                    self.stopping = False
                    break
        self.batchcurrent = None
        self.running = False
        self.stopping = False


    def getProgress(self):
        if(self.batchcurrent is not None):
            self.progress.run_progress = self.batchcurrent.workflow.getProgress()
        else:
            self.progress.run_progress = None
        return self.progress

        
    def stop(self):
        if(self.running):
            if(not self.stopping):
                # Cancel current batch
                self.stopping = True
                if(self.batchcurrent is not None):
                    self.batchcurrent.workflow.stop()
            elif(len(self.batchqueue) > 0):
                # Cancel next batch
                self.batchqueue.pop(0)


    def save(self, timestamp:int):
        save_file = f"{self.output_dir}/output_{timestamp}"
        rundata = self.rundata[timestamp]
        if(rundata.output is not None):
            if(isinstance(rundata.output, Image.Image)):
                rundata.output.save(f"{save_file}.png")
                rundata.save_file = f"{save_file}.png"
            elif(isinstance(rundata.output, Video)):
                shutil.copyfile(rundata.output.file.name, f"{save_file}.mp4")
                rundata.save_file = f"{save_file}.mp4"
            else:
                raise Exception("Output format not supported")

            file = open(f"{save_file}.yaml", "w")
            yaml.dump(rundata.params, file, width=float("inf"))
            print(rundata.params)
            print(f"Saved output to {save_file}")
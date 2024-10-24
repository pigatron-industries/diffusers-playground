from sympy import preview
from diffuserslib.functional import Video, Audio, FunctionalNode, WorkflowProgress
from typing import Any, Dict, Self, List, Callable
from dataclasses import dataclass, field
from PIL import Image
from PIL.Image import Resampling
from nicegui import run
import time
import datetime
import yaml
import copy
import traceback
import shutil
import os
import threading


@dataclass
class WorkflowRunData:
    timestamp:int
    params:Dict[str,Any]|None = None
    output: Any|None = None
    preview: Any|None = None
    save_file:str|None = None
    error:Exception|None = None
    start_time:datetime.datetime|None = None
    end_time:datetime.datetime|None = None
    duration:datetime.timedelta|None = None
    progress:WorkflowProgress = field(default_factory=lambda: WorkflowProgress(0, 0))

    def getStatus(self):
        if(self.error is not None):
            return "Error"
        elif(self.end_time is not None):
            return "Complete"
        elif self.start_time is not None and self.end_time is None:
            return "Running"
        else:
            return "Queued"


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
        self.batchrundata:Dict[int, WorkflowBatchData] = {} # Batches that have been run - includes currently running
        self.batchqueue:Dict[int, WorkflowBatchData] = {}   # Batch that have not been started yet
        self.batchcurrent:WorkflowBatchData|None = None     # Batch currently running
        self.rundatacurrent:WorkflowRunData|None = None  
        self.progress:BatchProgressData = BatchProgressData(0,0)
        self.stopping = False
        self.running = False
        self.thread = None
        self.batchcount = 0


    def getBatch(self, batchid:int):
        if(batchid in self.batchqueue):
            return self.batchqueue[batchid]
        elif(batchid in self.batchrundata):
            return self.batchrundata[batchid]
        else:
            return None
        

    def removeBatch(self, batchid:int):
        if(batchid in self.batchqueue):
            self.batchqueue.pop(batchid)
        elif(batchid in self.batchrundata):
            self.batchrundata.pop(batchid)
    

    def setWorkflow(self, workflow:FunctionalNode):
        self.workflow = workflow

    def clearRunData(self):
        self.rundata = {}
        self.batchrundata = {}


    def run(self, workflow:FunctionalNode, batch_size:int = 1):
        batchid = self.batchcount
        self.batchqueue[batchid] = WorkflowBatchData(batchid, copy.deepcopy(workflow), batch_size)
        self.batchcount += 1
        self.progress.jobs_remaining += batch_size
        if not self.running:
            self.thread = threading.Thread(target=self.process)
            self.thread.start()
        print(f"WorkflowRunner.run() {batchid}")
        return batchid
    

    def process(self):
        self.running = True
        while len(self.batchqueue) > 0:
            batchid, _ = list(self.batchqueue.items())[0]
            self.batchcurrent = self.batchqueue.pop(batchid)
            self.batchrundata[batchid] = self.batchcurrent
            self.batchcurrent.workflow.reset()
            print(f"Running workflow {self.batchcurrent.workflow.node_name} with batch size {self.batchcurrent.batch_size}")
            for i in range(self.batchcurrent.batch_size):
                print(f"Running workflow {self.batchcurrent.workflow.node_name} batch {i+1} of {self.batchcurrent.batch_size}")
                rundata = WorkflowRunData(int(time.time_ns()/1000))
                rundata.start_time = datetime.datetime.now()
                self.rundatacurrent = rundata
                self.batchcurrent.rundata[rundata.timestamp] = rundata
                self.rundata[rundata.timestamp] = rundata
                try:
                    self.batchcurrent.workflow.flush()
                    rundata.output = self.batchcurrent.workflow()
                except Exception as e:
                    rundata.error = e
                    print(f"Error running workflow {self.batchcurrent.workflow.node_name}: {e}")
                    traceback.print_exc()
                    break
                rundata.preview = self.createPreview(rundata.output)
                rundata.params = self.batchcurrent.workflow.getNodeOutputs()
                rundata.end_time = datetime.datetime.now()
                rundata.duration = rundata.end_time - rundata.start_time
                print(rundata.duration)
                self.progress.jobs_completed = len(self.rundata)
                self.progress.jobs_remaining = sum([batch.batch_size for batch in self.batchqueue.values()]) + self.batchcurrent.batch_size - len(self.batchcurrent.rundata)
                if(self.stopping == True):
                    self.stopping = False
                    break
        self.batchcurrent = None
        self.running = False
        self.stopping = False


    def getProgress(self):
        """ Updates the progress of the current batch """
        if(self.batchcurrent is not None):
            self.progress.run_progress = self.batchcurrent.workflow.getProgress()
            if(self.rundatacurrent):
                self.rundatacurrent.progress = self.progress.run_progress
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
                

    def createPreview(self, output:Any):
        if(isinstance(output, Image.Image)):
            preview = output.copy()
            preview.thumbnail((256,256), resample=Resampling.LANCZOS)
            return preview
        else:
            return None
        

    def save(self, timestamp:int, output_subdir:str|None = None):
        output_filename = f"output_{timestamp}"
        if(output_subdir is not None and len(output_subdir) > 0):
            output_dir = f"{self.output_dir}/{output_subdir}"
            try:
                os.makedirs(output_dir)
            except FileExistsError:
                pass
        else:
            output_dir = f"{self.output_dir}"
        save_file = f"{output_dir}/{output_filename}"
        rundata = self.rundata[timestamp]
        if(rundata.output is not None):
            if(isinstance(rundata.output, Image.Image)):
                rundata.output.save(f"{save_file}.png")
                rundata.save_file = f"{save_file}.png"
            elif(isinstance(rundata.output, Video)):
                filename = rundata.output.getFilename()
                assert filename is not None
                shutil.copyfile(filename, f"{save_file}.mp4")
                rundata.save_file = f"{save_file}.mp4"
            elif(isinstance(rundata.output, Audio)):
                filename = rundata.output.getFilename()
                assert filename is not None
                shutil.copyfile(filename, f"{save_file}.wav")
                rundata.save_file = f"{save_file}.wav"
            else:
                raise Exception("Output format not supported")

            file = open(f"{save_file}.yaml", "w")
            yaml.dump(rundata.params, file, width=float("inf"))
            print(rundata.params)
            print(f"Saved output to {save_file}")
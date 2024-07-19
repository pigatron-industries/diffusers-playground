from diffuserslib.functional.types import *

from diffuserslib.interface.WorkflowController import WorkflowController
from ..WorkflowComponents import WorkflowComponents
from .BatchRunDataControls import BatchRunDataControls
from nicegui import ui, run
from nicegui.element import Element
from typing import Dict, List
from dataclasses import dataclass
from PIL import Image



@dataclass
class BatchDataControls:
    batch_container:Element|None


class BatchInterfaceComponents(WorkflowComponents):

    def __init__(self, controller:WorkflowController):
        super().__init__(controller)
        self.rundata_controls:Dict[int, BatchRunDataControls] = {}
        self.batchdata_controls:Dict[int, BatchDataControls] = {}
        self.timer = ui.timer(1, lambda: self.updateWorkflowProgress(), active=False)
        self.progress = 0
        self.builders = {}
        for name, builder in self.controller.builders.items():
            if(builder.workflow):
                self.builders[name] = builder.name
        self.builders = dict(sorted(self.builders.items(), key=lambda item: item[1]))


    def runWorkflow(self):
        self.timer.activate()
        return self.controller.runWorkflow()


    def runSubWorkflow(self, workflow):
        self.timer.activate()
        return self.controller.runSubWorkflow(workflow)


    def updateWorkflowProgress(self):
        if(not self.controller.workflowrunner.running):
            self.timer.deactivate()
        self.status.refresh()
        for batchid, batchdata in self.controller.getBatchRunData().items():
            if(batchid in self.batchdata_controls):
                batchdata_container = self.batchdata_controls[batchid].batch_container
                for runid, rundata in batchdata.rundata.items():
                    if(runid in self.rundata_controls):
                        # Update existing output controls
                        self.rundata_controls[runid].update()
                    elif(batchdata_container is not None):
                        # Add new output controls
                        with batchdata_container:
                            self.workflow_output_rundata_container(batchid, runid)
            else:
                 with self.outputs_container:
                    self.workflow_output_batchdata(batchid) # type: ignore


    def clearOutputs(self):
        self.controller.workflowrunner.clearRunData()
        self.outputs.refresh()
        self.status.refresh()


    async def saveOutput(self, runid):
        result = await run.io_bound(self.controller.saveOutput, runid)
        filename = self.controller.getWorkflowRunData()[runid].save_file
        if (filename is not None):
            self.rundata_controls[runid].showLabelSaved(filename)


    def removeOutput(self, batchid, runid):
        self.controller.removeRunData(batchid, runid)
        self.status.refresh()
        self.outputs.refresh()
        # TODO don't refresh whole list - 
        # this should be changed to a dict[int, RunData] instead of a list so remove itesm doesn't re-index
        # self.rundata_controls[index].output_container.clear()
        # self.rundata_controls.pop(index)


    
    @ui.refreshable
    def status(self):
        with ui.column().classes("gap-0").style("align-items:center;"):  #margin-top:1.3em;
            if(self.controller.isStopping()):
                ui.label("Stopping...").style("color: #ff0000;")
            elif(self.controller.isRunning()):
                ui.label("Generating...").style("color: #ffcc00;")
            else:
                ui.label("Idle").style("color: #00ff00;")
            self.batch_queue_progress()

    
    def batch_queue_progress(self):
        donebatches = self.controller.getBatchRunData()
        currentbatch = self.controller.getBatchCurrentData()
        queuebatches = self.controller.getBatchQueueData()
        with ui.row():
            for batchid, batch in donebatches.items():
                if(batch != currentbatch):
                    ui.button(f"{len(batch.rundata)}", color="slate").classes('px-2 text-green-500').props('dense')
            if(currentbatch is not None):
                currentbatch_done = len(currentbatch.rundata)
                currentbatch_remaining = currentbatch.batch_size - currentbatch_done
                ui.button(f"{currentbatch_done} - {currentbatch_remaining}", color="slate").classes('px-2 text-yellow-500').props('dense')
            for batch in queuebatches:
                ui.button(f"{batch.batch_size}", color="slate").classes('px-2 text-gray-500').props('dense')

    
    def buttons(self):
        ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size').bind_visibility_from(self.controller.model, 'run_type', value=1).style("width: 100px")
        ui.button('Run', on_click=lambda e: self.runWorkflow()).classes('align-middle')
        ui.button('Stop', on_click=lambda e: self.controller.stopWorkflow()).classes('align-middle')
        ui.button('Clear', on_click=lambda e: self.clearOutputs()).classes('align-middle')


    def settings(self):
        ui.input(label='Output Sub-Directory').bind_value(self.controller, 'output_subdir')


    @ui.refreshable
    def controls(self):
        self.workflowSelect(self.builders, ["Image", "Video", "Audio", "str", "Other"])
        self.workflow_parameters()


    @ui.refreshable
    def outputs(self):
        self.rundata_controls = {}
        self.outputs_container = ui.column().classes('w-full')
        with self.outputs_container:
            for batchid, batchdata in self.controller.getBatchRunData().items():
                self.workflow_output_batchdata(batchid)


    def workflow_output_batchdata(self, batchid):
        batchdata = self.controller.getBatchRunData()[batchid]
        batchdata_container = ui.column().classes('w-full p-2')
        with batchdata_container:
            ui.label(f"Batch {batchid}")
            for runid, rundata in batchdata.rundata.items():
                self.workflow_output_rundata_container(batchid, runid)
        self.batchdata_controls[batchid] = BatchDataControls(batchdata_container)


    def workflow_output_rundata_container(self, batchid, runid):
        rundata_container = ui.card_section().classes('w-full').style("background-color:#2b323b; border-radius:8px;")
        with rundata_container:
            self.rundata_controls[runid] = BatchRunDataControls(self.controller.getWorkflowRunData()[runid], runid, batchid, rundata_container, self.controller)


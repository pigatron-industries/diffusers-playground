from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes.user import UserInputNode, ListUserInputNode
from diffuserslib.functional.nodes.animated import Video
from .Controller import Controller
from .InterfaceComponents import InterfaceComponents
from nicegui import ui, run
from nicegui.element import Element
from nicegui.elements.label import Label
from typing import Dict, List
from dataclasses import dataclass
from PIL import Image


default_output_width = 256


@dataclass
class RunDataControls:
    rundata_container:Element|None
    output_control:Element|None
    output_width:int
    label_saved:Label
    waiting_output:bool = False
    expanded:bool = False

    def showLabelSaved(self, filename:str):
        self.label_saved.set_text(f"Saved to {filename}")
        self.label_saved.set_visibility(True)

    def toggleExpanded(self):
        self.expanded = not self.expanded
        if(self.output_control is not None):
            if(self.expanded):
                self.output_control.style(replace= f"max-width:{self.output_width}px; min-width:{self.output_width}px;")
            else:
                self.output_control.style(replace = f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
            

@dataclass
class BatchDataControls:
    batch_container:Element|None


class BatchInterfaceComponents(InterfaceComponents):

    def __init__(self, controller:Controller):
        super().__init__(controller)
        self.rundata_controls:Dict[int, RunDataControls] = {}
        self.batchdata_controls:Dict[int, BatchDataControls] = {}
        self.timer = ui.timer(1, lambda: self.updateWorkflowProgress(), active=False)


    async def runWorkflow(self):
        self.timer.activate()
        result = await run.io_bound(self.controller.runWorkflow)


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
                        rundata_container = self.rundata_controls[runid].rundata_container
                        if(rundata.output is not None and self.rundata_controls[runid].waiting_output == True and rundata_container is not None):
                            rundata_container.clear()
                            with rundata_container:
                                self.rundata_controls[runid].output_control, self.rundata_controls[runid].output_width, self.rundata_controls[runid].label_saved, self.rundata_controls[runid].waiting_output = self.workflow_output_rundata(batchid, runid)
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


    def saveOutput(self, key):
        self.controller.saveOutput(key)
        filename = self.controller.getWorkflowRunData()[key].save_file
        if (filename is not None):
            self.rundata_controls[key].showLabelSaved(filename)


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


    @ui.refreshable
    def controls(self):
        self.workflowSelect(self.controller.workflows_batch)
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
            output_control, output_width, label_saved, waiting_output = self.workflow_output_rundata(batchid=batchid, runid=runid)
            self.rundata_controls[runid] = RunDataControls(rundata_container, output_control, output_width, label_saved, waiting_output)


    def workflow_output_rundata(self, batchid, runid):
        rundata = self.controller.getWorkflowRunData()[runid]
        with ui.row().classes('w-full no-wrap'):
            output_control = None
            output_width = 0
            waiting_output = False

            if(rundata.error is not None):
                output_control = ui.label(f"Error: {rundata.error}").style("color: #ff0000;")
            elif(rundata.output is None):
                output_control = ui.label("Generating...").style(replace= f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
                waiting_output = True
            elif(isinstance(rundata.output, Image.Image)):
                output_width = rundata.output.width
                output_control = ui.image(rundata.output).on('click', lambda e: self.rundata_controls[runid].toggleExpanded()).style(f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
            elif(isinstance(rundata.output, Video)):
                output_control = ui.video(rundata.output.file.name).style(replace= f"max-width:{default_output_width}px; min-width:{default_output_width}px;")

            with ui.column():
                if rundata.params is not None:
                    for node_name in rundata.params:
                        for paramname, paramvalue in rundata.params[node_name].items():
                            with ui.row().classes('no-wrap'):
                                ui.label(f"{paramname}:").style("line-height: 1;")
                                if(isinstance(paramvalue, Image.Image)):
                                    ui.image(paramvalue).style("min-width:128px; min-height:128px;")
                                else:
                                    ui.label(str(paramvalue)).style("line-height: 1;")

            with ui.column().classes('ml-auto'):
                with ui.row().classes('ml-auto'):
                    ui.button('Save', on_click=lambda e: self.saveOutput(runid))
                    ui.button('Remove', on_click=lambda e: self.removeOutput(batchid, runid))      
                with ui.row():
                    label_saved = ui.label(f"Saved to {rundata.save_file}")
                    label_saved.set_visibility(rundata.save_file is not None)
        return output_control, output_width, label_saved, waiting_output
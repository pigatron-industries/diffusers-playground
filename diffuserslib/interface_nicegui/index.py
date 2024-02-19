from nicegui import ui, context, run
from nicegui.elements.label import Label
from nicegui.element import Element
from .api import *
from .Controller import Controller
from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from diffuserslib.functional.nodes.user.ListUserInputNode import ListUserInputNode
from diffuserslib.functional.WorkflowRunner import WorkflowRunData
from typing import List, Dict
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


class View:

    @ui.page('/')
    def gui():
        app.on_exception(ui.notify)
        view = View()
        view.page()


    def __init__(self):
        self.controller = Controller()
        self.rundata_controls:Dict[int, RunDataControls] = {}
        self.batchdata_controls:Dict[int, BatchDataControls] = {}


    def loadWorkflow(self, workflow_name):
        print("loading workflow")
        self.controller.loadWorkflow(workflow_name)
        self.workflow_controls.refresh()


    def setParam(self, node_name, param_name, value, index=None):
        self.controller.setParam(node_name, param_name, value, index)


    async def runWorkflow(self):
        self.timer.activate()
        result = await run.io_bound(self.controller.runWorkflow)


    def stopWorkflow(self):
        self.controller.workflowrunner.stop()


    def clearOutputs(self):
        self.controller.workflowrunner.clearRunData()
        self.workflow_outputs.refresh()


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


    def getWorkflowRunData(self) -> Dict[int, WorkflowRunData]:
        return self.controller.getWorkflowRunData()
    

    def expandOutput(self, index):
        self.rundata_controls[index].toggleExpanded()


    def saveOutput(self, key):
        self.controller.saveOutput(key)
        filename = self.getWorkflowRunData()[key].save_file
        if (filename is not None):
            self.rundata_controls[key].showLabelSaved(filename)


    def removeOutput(self, batchid, runid):
        self.controller.removeRunData(batchid, runid)
        self.status.refresh()
        self.workflow_outputs.refresh()

        # TODO don't refresh whole list - 
        # this should be changed to a dict[int, RunData] instead of a list so remove itesm doesn't re-index
        # self.rundata_controls[index].output_container.clear()
        # self.rundata_controls.pop(index)


    def toggleParamFunctional(self, param:NodeParameter):
        if(isinstance(param.value, UserInputNode)):
            param.value = FunctionalNode("empty")
        else:
            param.value = param.initial_value
        self.workflow_controls.refresh()


    def selectInputNode(self, param:NodeParameter, value):
        self.controller.createInputNode(param, value)
        self.workflow_controls.refresh()


    def page(self):
        ui.page_title('Generative Toolkit')
        ui.label.default_classes('label')
        ui.select.default_classes('w-80')
        ui.query('body').style(f'background-color: rgb(24, 28, 33)')
        ui.add_head_html('''
            <style>
                .align-middle {
                    transform: translateY(50%);
                }
            </style>
        ''')
        context.get_client().content.classes('h-[100vh] p-0')
        self.timer = ui.timer(1, lambda: self.updateWorkflowProgress(), active=False)

        with ui.column().classes("w-full h-full no-wrap gap-0"):
            with ui.row().classes("w-full p-2 place-content-between").style("background-color:#2b323b; border-bottom:1px solid #585b5f"):
                ui.toggle({1: 'Batch', 2: 'Animate', 3: 'Realtime'}).bind_value(self.controller.model, 'run_type').style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')
                self.status() # type: ignore
                with ui.row():
                    ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size').bind_visibility_from(self.controller.model, 'run_type', value=1).style("width: 100px")
                    ui.button('Run', on_click=lambda e: self.runWorkflow()).classes('align-middle')
                    ui.button('Stop', on_click=lambda e: self.controller.stopWorkflow()).classes('align-middle')
                    ui.button('Clear', on_click=lambda e: self.clearOutputs()).classes('align-middle')
            with ui.splitter(value=40).classes("w-full h-full no-wrap overflow-auto") as splitter:
                with splitter.before:
                    with ui.column().classes("p-2 w-full"):
                        ui.select(list(self.controller.workflows.keys()), value=self.controller.model.workflow_name, label='Workflow', on_change=lambda e: self.loadWorkflow(e.value))
                        self.workflow_controls() # type: ignore
                with splitter.after:
                    self.workflow_outputs() # type: ignore


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


    @ui.refreshable
    def workflow_controls(self):
        if self.controller.workflow is not None:
            with ui.card_section().classes('w-full').style("background-color:rgba(255, 255, 255, 0.1); border-radius:8px;"):
                with ui.column():
                    self.node_parameters(self.controller.workflow)


    def node_parameters(self, node:FunctionalNode):
        params = node.getParams()
        for param in params:
            if(isinstance(param.initial_value, UserInputNode)):
                with ui.row().classes('w-full'):
                    self.workflow_parameter(param)
            elif(isinstance(param.value, FunctionalNode)):
                self.node_parameters(param.value)


    def workflow_parameter(self, param:NodeParameter):
        input_nodes = self.controller.getSelectableInputNodes(param)
        if(len(input_nodes) > 0):
            ui.button(icon='functions', color='dark', on_click=lambda e: self.toggleParamFunctional(param)).classes('align-middle').props('dense')
        else:
            ui.label().classes('w-8')
        if(isinstance(param.value, ListUserInputNode)):
            param.value.ui(child_renderer=self.node_parameters) # type: ignore
        elif(isinstance(param.value, UserInputNode)):
            param.value.ui()
        else:
            with ui.card_section().classes('grow').style("background-color:rgba(255, 255, 255, 0.1); border-radius:8px;"):
                with ui.column():
                    selected_node = param.value.node_name if param.value.node_name != "empty" else None
                    ui.select(input_nodes, value=selected_node, label=param.name, on_change=lambda e: self.selectInputNode(param, e.value))
                    self.node_parameters(param.value)
            

    @ui.refreshable
    def workflow_outputs(self):
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
        rundata = self.getWorkflowRunData()[runid]
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
                output_control = ui.image(rundata.output).on('click', lambda e: self.expandOutput(runid)).style(f"max-width:{default_output_width}px; min-width:{default_output_width}px;")

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
    
            
from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from diffuserslib.functional.nodes.user.ListUserInputNode import ListUserInputNode
from .Controller import Controller
from .InterfaceComponents import InterfaceComponents
from nicegui import ui, run
from nicegui.element import Element
from nicegui.elements.label import Label
from typing import Dict
from dataclasses import dataclass
from PIL import Image


default_output_width = 256


class RealtimeInterfaceComponents(InterfaceComponents):

    def __init__(self, controller:Controller):
        super().__init__(controller)
        self.timer = ui.timer(0.1, lambda: self.updateWorkflowOutput(), active=False)


    def loadWorkflow(self, workflow_name):
        print("loading workflow")
        self.controller.loadWorkflow(workflow_name)
        self.controls.refresh()


    async def runWorkflow(self):
        self.timer.activate()


    def toggleParamFunctional(self, param:NodeParameter):
        if(isinstance(param.value, UserInputNode)):
            param.value = FunctionalNode("empty")
        else:
            param.value = param.initial_value
        self.controls.refresh()


    def selectInputNode(self, param:NodeParameter, value):
        self.controller.createInputNode(param, value)
        self.controls.refresh()


    def updateWorkflowOutput(self):
        if(self.controller.workflow is not None):
            self.controller.workflow.flush()
            output = self.controller.workflow()
            # TODO show output


    def clearOutputs(self):
        self.controller.workflowrunner.clearRunData()
        self.outputs.refresh()
        self.status.refresh()

    
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
        ui.select(list(self.controller.workflows_batch.keys()), value=self.controller.model.workflow_name, label='Workflow', on_change=lambda e: self.loadWorkflow(e.value))
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
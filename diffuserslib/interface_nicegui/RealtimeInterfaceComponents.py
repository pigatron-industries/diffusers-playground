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
        self.output = None
        self.timer = ui.timer(0.1, lambda: self.updateWorkflowOutput(), active=False)


    def runWorkflow(self):
        self.timer.activate()

    def stopWorkflow(self):
        self.timer.deactivate()


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
            self.output = output
            self.output_control.style(replace = f"max-width:{output.width}px; max-height:{output.height}px;")
            self.output_control.set_source(output)


    def resetWorkflow(self):
        if(self.controller.workflow is not None):
            self.controller.workflow.reset()
            self.outputs.refresh()
            self.status.refresh()

    
    @ui.refreshable
    def status(self):
        with ui.column().classes("gap-0").style("align-items:center;"):  #margin-top:1.3em;
            if(self.controller.isRunning()):
                ui.label("Running").style("color: #ffcc00;")
            else:
                ui.label("Idle").style("color: #00ff00;")

    
    def buttons(self):
        ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size').bind_visibility_from(self.controller.model, 'run_type', value=1).style("width: 100px")
        ui.button('Run', on_click=lambda e: self.runWorkflow()).classes('align-middle')
        ui.button('Stop', on_click=lambda e: self.stopWorkflow()).classes('align-middle')
        ui.button('Reset', on_click=lambda e: self.resetWorkflow()).classes('align-middle')


    @ui.refreshable
    def controls(self):
        self.workflowSelect(self.controller.workflows_realtime)
        self.workflow_parameters()


    @ui.refreshable
    def outputs(self):
        self.rundata_controls = {}
        self.output_container = ui.column().classes('w-full')
        with self.output_container:
            with ui.card_section().classes('w-full').style("background-color:#2b323b; border-radius:8px;"):
                if(self.output is not None):
                    self.output_control = ui.interactive_image(self.output).style(replace = f"max-width:{self.output.width}px; max-height:{self.output.height}px;")
                else:
                    self.output_control = ui.interactive_image()
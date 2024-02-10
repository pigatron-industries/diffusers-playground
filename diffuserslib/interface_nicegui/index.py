from nicegui import ui, context
from nicegui.elements.label import Label
from nicegui.element import Element
from .api import *
from .Controller import Controller
from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.FunctionalTyping import ParamType
from diffuserslib.functional.WorkflowRunner import WorkflowRunData
from typing import List
from dataclasses import dataclass
from PIL import Image


default_image_class = 'w-64'


@dataclass
class OutputControls:
    output_control:Element|None
    output_width:int
    label_saved:Label
    expanded:bool = False

    def showLabelSaved(self, filename:str):
        self.label_saved.set_text(f"Saved to {filename}")
        self.label_saved.set_visibility(True)

    def toggleExpanded(self):
        self.expanded = not self.expanded
        if(self.output_control is not None):
            if(self.expanded):
                self.output_control.classes(remove = default_image_class)
            else:
                self.output_control.classes(add = default_image_class)
            


class View:

    @ui.page('/')
    def gui():
        view = View()
        view.page()

    def __init__(self):
        self.controller = Controller()
        self.output_controls:List[OutputControls] = []

    def loadWorkflow(self, workflow_name):
        print("loading workflow")
        self.controller.loadWorkflow(workflow_name)
        self.workflow_controls.refresh()

    def setParam(self, node_name, param_name, value):
        self.controller.setParam(node_name, param_name, value)

    def runWorkflow(self):
        self.controller.runWorkflow()
        self.workflow_outputs.refresh()

    def getWorkflowRunData(self) -> List[WorkflowRunData]:
        return self.controller.workflowrunner.rundata
    
    def expandOutput(self, index):
        self.output_controls[index].toggleExpanded()

    def saveOutput(self, index):
        # TODO actually save the file
        self.getWorkflowRunData()[index].save_file = "test/file/blah/blah/path.png"
        self.output_controls[index].showLabelSaved(self.getWorkflowRunData()[index].save_file)

    def removeOutput(self, index):
        self.getWorkflowRunData().pop(index)
        self.workflow_outputs.refresh()


    def page(self):
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


        context.get_client().content.classes('h-[100vh]')
        with ui.column().classes("w-full h-full no-wrap").style("height: 100%"):
            with ui.splitter(value=30).classes("w-full h-full no-wrap").style("height: 100%") as splitter:
                with splitter.before:
                    with ui.row():
                        batch_size = ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size')
                        run_button = ui.button('Run', on_click=lambda e: self.runWorkflow()).classes('align-middle')
                    with ui.row():
                        workflow = ui.select(list(self.controller.workflows.keys()), value=self.controller.model.workflow_name, label='Workflow', on_change=lambda e: self.loadWorkflow(e.value))
                    self.workflow_controls()
                with splitter.after:
                    self.workflow_outputs()


    @ui.refreshable
    def workflow_controls(self):
        if self.controller.workflow is not None:
            paramInfo = self.controller.workflow.getStaticParams()
            for node_name in paramInfo.params:
                with ui.card_section():
                    ui.label(node_name)
                    for param in paramInfo.params[node_name]:
                        with ui.row():
                            self.workflow_parameter(node_name, param)

    def workflow_parameter(self, node_name, param):
        match param.type.type:
            case ParamType.INT:
                ui.number(value=param.value, label=param.name, on_change=lambda e: self.setParam(node_name, param.name, int(e.value)))
            case ParamType.FLOAT:
                ui.number(value=param.value, label=param.name, format='%.2f', on_change=lambda e: self.setParam(node_name, param.name, e.value))
            case ParamType.STRING:
                ui.input(value=param.value, label=param.name, on_change=lambda e: self.setParam(node_name, param.name, e.value))
            case ParamType.BOOL:
                if(param.type.size == 1 or param.type.size is None):
                    ui.switch(value=param.value, on_change=lambda e: self.setParam(node_name, param.name, e.value))
                else:
                    for i in range(param.type.size):
                        ui.switch(value=param.value[i], on_change=lambda e: self.setParam(node_name, param.name, e.value))
            case ParamType.IMAGE_SIZE:
                ui.number(value=param.value[0], label=f"{param.name} width", on_change=lambda e: self.setParam(node_name, param.name, (e.value, param.value[1])))
                ui.number(value=param.value[1], label=f"{param.name} height", on_change=lambda e: self.setParam(node_name, param.name, (param.value[0], e.value)))
            case ParamType.COLOUR:
                ui.color_input(value=param.value, label=param.name, on_change=lambda e: self.setParam(node_name, param.name, e.value))
            case _:
                ui.label(param.name)

    @ui.refreshable
    def workflow_outputs(self):
        self.output_controls = []
        with ui.column().classes('w-full pl-5'):
            for i, rundata in enumerate(self.getWorkflowRunData()):
                self.workflow_output(i)
                

    @ui.refreshable
    def workflow_output(self, index):
        rundata = self.getWorkflowRunData()[index]
        with ui.card_section().classes('w-full').style("background-color:rgb(43, 50, 59); border-radius:8px;"):
                with ui.row().classes('w-full'):
                    output_control = None
                    output_width = 0
                    if(isinstance(rundata.output, Image.Image)):
                        output_width = rundata.output.width
                        output_control = ui.image(rundata.output).on('click', lambda e: self.expandOutput(index)).classes(default_image_class).style(f"max-width: {output_width}px;")
                    with ui.column():
                        for node_name in rundata.params.params:
                            for param in rundata.params.params[node_name]:
                                ui.label(f"{param.name}: {param.value}").style("line-height: 1;")
                    with ui.column().classes('ml-auto'):
                        with ui.row().classes('ml-auto'):
                            ui.button('Save', on_click=lambda e: self.saveOutput(index))
                            ui.button('Remove', on_click=lambda e: self.removeOutput(index))      
                        with ui.row():
                            label_saved = ui.label(f"Saved to {rundata.save_file}")
                            label_saved.set_visibility(rundata.save_file is not None)
        self.output_controls.append(OutputControls(output_control, output_width, label_saved))
            
from nicegui import ui, context, run
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


default_output_class = 'w-64'


@dataclass
class OutputControls:
    output_container:Element|None
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
                self.output_control.classes(remove = default_output_class)
            else:
                self.output_control.classes(add = default_output_class)
            


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
        for i, rundata in enumerate(self.getWorkflowRunData()):
            if(i < len(self.output_controls)):
                # Update existing output controls
                output_container = self.output_controls[i].output_container
                if(rundata.output is not None and self.output_controls[i].waiting_output == True and output_container is not None):
                    output_container.clear()
                    with output_container:
                        self.output_controls[i].output_control, self.output_controls[i].output_width, self.output_controls[i].label_saved, self.output_controls[i].waiting_output = self.workflow_output(i)
            else:
                # Add new output controls
                with self.outputs_container:
                    output_container = ui.card_section().classes('w-full').style("background-color:rgb(43, 50, 59); border-radius:8px;")
                    with output_container:
                        output_control, output_width, label_saved, waiting_output = self.workflow_output(i)
                        self.output_controls.append(OutputControls(output_container, output_control, output_width, label_saved, waiting_output))

                


        for i, output_controls in enumerate(self.output_controls):
            if(self.getWorkflowRunData()[i].output is not None and output_controls.output_control is None and output_controls.output_container is not None):
                output_controls.output_container.clear()
                with output_controls.output_container:
                    output_controls.output_control, output_controls.output_width, output_controls.label_saved = self.workflow_output(i)


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
        # TODO only remove elements that are no longer in the list
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
        context.get_client().content.classes('h-[100vh] p-0')
        self.timer = ui.timer(1, lambda: self.updateWorkflowProgress(), active=False)

        with ui.column().classes("w-full h-full no-wrap gap-0"):
            with ui.row().classes("w-full p-2 place-content-between").style("background-color:#2b323b; border-bottom:1px solid #585b5f"):
                ui.toggle({1: 'Batch', 2: 'Animate', 3: 'Realtime'}).bind_value(self.controller.model, 'run_type').style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')

                with ui.row():
                    ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size').bind_visibility_from(self.controller.model, 'run_type', value=1).style("width: 100px")
                    ui.button('Run', on_click=lambda e: self.runWorkflow()).classes('align-middle')
                    ui.button('Stop', on_click=lambda e: self.stopWorkflow()).classes('align-middle')
                    ui.button('Clear', on_click=lambda e: self.clearOutputs()).classes('align-middle')
            with ui.splitter(value=30).classes("w-full h-full no-wrap overflow-auto") as splitter:
                with splitter.before:
                    with ui.column().classes("p-2"):
                        ui.select(list(self.controller.workflows.keys()), value=self.controller.model.workflow_name, label='Workflow', on_change=lambda e: self.loadWorkflow(e.value))
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
        self.outputs_container = ui.column().classes('w-full p-2')
        with self.outputs_container:
            for i, rundata in enumerate(self.getWorkflowRunData()):
                output_container = ui.card_section().classes('w-full').style("background-color:#2b323b; border-radius:8px;")
                with output_container:
                    output_control, output_width, label_saved, waiting_output = self.workflow_output(i)
                    self.output_controls.append(OutputControls(output_container, output_control, output_width, label_saved, waiting_output))


    @ui.refreshable
    def workflow_output(self, index):
        rundata = self.getWorkflowRunData()[index]
        with ui.row().classes('w-full'):
            output_control = None
            output_width = 0
            waiting_output = False
            if(rundata.output is None):
                output_control = ui.label("Generating...").classes(default_output_class)
                waiting_output = True
            if(isinstance(rundata.output, Image.Image)):
                output_width = rundata.output.width
                output_control = ui.image(rundata.output).on('click', lambda e: self.expandOutput(index)).classes(default_output_class).style(f"max-width: {output_width}px;")
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
        return output_control, output_width, label_saved, waiting_output
    
            
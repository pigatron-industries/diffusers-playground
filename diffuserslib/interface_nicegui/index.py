from nicegui import ui, context
from .api import *
from .Controller import Controller
from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.FunctionalTyping import ParamType
from diffuserslib.functional.WorkflowRunner import WorkflowRunData
from typing import Self


class View:

    def __init__(self):
        self.controller = Controller()
        # self.model = Model()

    def loadWorkflow(self, workflow_name):
        print("loading workflow")
        self.controller.loadWorkflow(workflow_name)
        View.workflow_controls.refresh()

    def setParam(self, node_name, param_name, value):
        self.controller.setParam(node_name, param_name, value)

    def runWorkflow(self):
        self.controller.runWorkflow()
        View.workflow_output.refresh()

    def getWorkflowRunData(self) -> list[WorkflowRunData]:
        return self.controller.workflowrunner.rundata

    @ui.page('/')
    def gui():
        view = View()

        ui.add_head_html('''
            <style>
                .label {
                    transform: translateY(50%);
                }
            </style>
        ''')
        ui.label.default_classes('label')
        ui.select.default_classes('w-80')
        ui.query('body').style(f'background-color: rgb(24, 28, 33)')


        context.get_client().content.classes('h-[100vh]')
        with ui.column().classes("w-full h-full no-wrap").style("height: 100%"):
            with ui.splitter(value=30).classes("w-full h-full no-wrap").style("height: 100%") as splitter:
                with splitter.before:
                    with ui.row():
                        batch_size = ui.number(label="Batch Size").bind_value(view.controller, 'batch_size')
                        run_button = ui.button('Run', on_click=lambda e: view.runWorkflow()).classes('label')
                    with ui.row():
                        workflow = ui.select(list(view.controller.workflows.keys()), label='Workflow', on_change=lambda e: view.loadWorkflow(e.value))
                    View.workflow_controls(view)
                with splitter.after:
                    View.workflow_output(view)


    @staticmethod
    @ui.refreshable
    def workflow_controls(view):
        if view.controller.workflow is not None:
            paramInfo = view.controller.workflow.getStaticParams()
            for node_name in paramInfo.params:
                with ui.card_section():
                    print(node_name)
                    ui.label(node_name)
                    for param in paramInfo.params[node_name]:
                        with ui.row():
                            ui.row()
                            match param.type.type:
                                case ParamType.INT:
                                    ui.number(value=param.value, label=param.name, on_change=lambda e: view.setParam(node_name, param.name, e.value))
                                case ParamType.FLOAT:
                                    ui.number(value=param.value, label=param.name, format='%.2f', on_change=lambda e: view.setParam(node_name, param.name, e.value))
                                case ParamType.STRING:
                                    ui.input(value=param.value, label=param.name, on_change=lambda e: view.setParam(node_name, param.name, e.value))
                                case ParamType.BOOL:
                                    if(param.type.size == 1 or param.type.size is None):
                                        ui.switch(value=param.value, on_change=lambda e: view.setParam(node_name, param.name, e.value))
                                    else:
                                        for i in range(param.type.size):
                                            ui.switch(value=param.value[i], on_change=lambda e: view.setParam(node_name, param.name, e.value))
                                case ParamType.IMAGE_SIZE:
                                    ui.number(value=param.value[0], label=f"{param.name} width", on_change=lambda e: view.setParam(node_name, param.name, (e.value, param.value[1])))
                                    ui.number(value=param.value[1], label=f"{param.name} height", on_change=lambda e: view.setParam(node_name, param.name, (param.value[0], e.value)))
                                case ParamType.COLOUR:
                                    ui.color_input(value=param.value, label=param.name, on_change=lambda e: view.setParam(node_name, param.name, e.value))
                                case _:
                                    ui.label(param.name)


    @staticmethod
    @ui.refreshable
    def workflow_output(view):
        for rundata in view.getWorkflowRunData():
            with ui.card_section():
                print("render image")
                ui.label("image")
                ui.image(rundata.output).classes('w-32')
                
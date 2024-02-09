from nicegui import ui, context
from .api import *
from ..functional_nodes.FunctionalNode import FunctionalNode


class Model:
    def __init__(self):
        self.workflows = None
        self.workflow:FunctionalNode|None = None


class Controller:
    def __init__(self):
        self.model = Model()
        self.model.workflows = loadWorkflows()


    def loadWorkflow(self, workflow_name):
        self.model.workflow = self.model.workflows[workflow_name].build()
        Interface.workflow_controls.refresh()


class Interface:

    @ui.page('/')
    def gui():
        controller = Controller()

        ui.add_head_html('''
            <style>
                .label {
                    transform: translateY(50%);
                }
            </style>
        ''')
        ui.button.default_props('dense')
        ui.select.default_props('dense')
        ui.input.default_props('dense')
        ui.label.default_classes('label')

        context.get_client().content.classes('h-[100vh]')
        with ui.column().classes("w-full h-full no-wrap").style("height: 100%"):
            with ui.splitter().classes("w-full h-full no-wrap").style("height: 100%") as splitter:
                with splitter.before:
                    with ui.row():
                        ui.label('Workflow')
                        workflow = ui.select(list(controller.model.workflows.keys()), on_change=lambda e: controller.loadWorkflow(e.value))
                    Interface.workflow_controls(controller)
                with splitter.after:
                    ui.label('This is some content on the right hand side.').classes('ml-2')


    @staticmethod
    @ui.refreshable
    def workflow_controls(controller):
        if controller.model.workflow is not None:
            paramInfo = controller.model.workflow.getStaticParams()
            for node_name in paramInfo.params:
                with ui.card_section():
                    print(node_name)
                    ui.label(node_name)
                    for param in paramInfo.params[node_name]:
                        with ui.row():
                            ui.label(param.name)
                            ui.label(param.value)

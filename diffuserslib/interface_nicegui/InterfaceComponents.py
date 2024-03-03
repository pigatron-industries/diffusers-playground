from .Controller import Controller
from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes import UserInputNode, ListUserInputNode
from nicegui import ui
from typing import Dict, List


class InterfaceComponents:

    def __init__(self, controller:Controller):
        self.controller = controller


    def loadWorkflow(self, workflow_name):
        print("loading workflow")
        self.controller.loadWorkflow(workflow_name)
        self.controls.refresh()


    def toggleParamFunctional(self, param:NodeParameter):
        if(isinstance(param.value, UserInputNode)):
            param.value = FunctionalNode("empty")
        else:
            param.value = param.initial_value
        self.controls.refresh()


    def selectInputNode(self, param:NodeParameter, value):
        self.controller.createInputNode(param, value)
        self.controls.refresh()


    def status(self):
        pass

    def buttons(self):
        pass

    def settings(self):
        pass

    def controls(self):
        pass

    def outputs(self):
        pass

    
    def workflowSelect(self, workflow_list:Dict[str, str]):
        if(self.controller.model.workflow_name not in workflow_list):
            self.controller.model.workflow_name = None
            self.controller.workflow = None
        ui.select(workflow_list, value=self.controller.model.workflow_name, label='Workflow', on_change=lambda e: self.loadWorkflow(e.value))


    def workflow_parameters(self):
        self.input_nodes = []
        if self.controller.workflow is not None:
            with ui.card_section().classes('w-full').style("background-color:rgba(255, 255, 255, 0.1); border-radius:8px;"):
                with ui.column():
                    self.node_parameters(self.controller.workflow)


    def node_parameters(self, node:FunctionalNode):
        params = node.getInitParams() + node.getParams()
        for param in params:
            if(isinstance(param.initial_value, UserInputNode) and param.initial_value not in self.input_nodes):
                self.input_nodes.append(param.initial_value)
                with ui.row().classes('w-full'):
                    self.node_parameter(param)
            elif(isinstance(param.value, FunctionalNode)):
                self.node_parameters(param.value)
            elif(isinstance(param.value, List)):
                for item in param.value:
                    if(isinstance(item, FunctionalNode)):
                        self.node_parameters(item)


    def node_parameter(self, param:NodeParameter):
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
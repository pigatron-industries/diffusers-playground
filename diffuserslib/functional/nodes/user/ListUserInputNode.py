from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from .UserInputNode import UserInputNode
from nicegui import ui
import copy

class ListUserInputNode(UserInputNode):
    """ user interface node to supply a variable size list of items """
    def __init__(self, input_node_generator:Callable[[], FunctionalNode], name:str="list_user_input"):
        self.input_node_generator = input_node_generator
        self.value:List[Any] = []
        super().__init__(name)
        self.addParam("list", self.value, List[Any])


    def getValue(self) -> int:
        return len(self.value)
    

    def setValue(self, value:int):
        self.value = [self.input_node_generator() for i in range(value)]


    @ui.refreshable
    def ui(self, child_renderer:Callable[[FunctionalNode], None]):
        with ui.column():
            ui.label(f"{self.node_name}")
            ui.button(icon="add", on_click = lambda e: self.addInput(0))
        for i, item in enumerate(self.value):
            with ui.row().classes('w-full'):
                ui.label().classes('w-8')
                with ui.card_section().classes('grow').style("background-color:rgba(255, 255, 255, 0.1); border-radius:8px;"):
                    with ui.column():
                        # TODO find child user input node and call its ui method
                        # ui.label(f"Item {i+1}").classes('mr-2')
                        child_renderer(item)
                    self.removeButton(i)


    def removeButton(self, index):
        ui.button(icon='remove', on_click = lambda e: self.removeInput(index))


    def node_parameters(self, node:FunctionalNode):
        params = node.getParams()
        for param in params:
            if(isinstance(param.initial_value, UserInputNode)):
                with ui.row().classes('w-full'):
                    self.workflow_parameter(param)
            elif(isinstance(param.value, FunctionalNode)):
                self.node_parameters(param.value)


    def addInput(self, index:int):
        newvalue = self.input_node_generator()
        self.value.insert(index, newvalue)
        self.ui.refresh()


    def removeInput(self, index:int):
        self.value.pop(index)
        self.ui.refresh()


    # def process(self, list:List[Any]) -> List[Any]:
    #     return list
    
    def process(self, **kwargs) -> List[Any]:
        return self.value
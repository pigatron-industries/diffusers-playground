from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .UserInputNode import UserInputNode
from nicegui import ui


class ListUserInputNode(UserInputNode):
    """ user interface node to supply a variable size list of items """
    def __init__(self, input_node_generator:Callable[[], FunctionalNode], name:str="list_user_input"):
        self.input_node_generator = input_node_generator
        super().__init__(name)
        self.addParam("list", [], List[Any])


    def getValue(self) -> int:
        return len(self.params["list"].value)
    

    def setValue(self, value:int):
        self.params["list"].value = [self.input_node_generator() for i in range(value)]


    @ui.refreshable
    def gui(self, child_renderer:Callable[[FunctionalNode], None]):
        with ui.row():
            ui.button(icon="add", on_click = lambda e: self.addInput(0)).props('dense')
            ui.label(f"{self.node_name}")
        for i, item in enumerate(self.params["list"].value):
            with ui.row().classes('w-full'):
                ui.label().classes('w-8')
                with ui.card_section().classes('grow').style("background-color:rgba(255, 255, 255, 0.1); border-radius:8px;"):
                    with ui.column():
                        child_renderer(item)
                    self.removeButton(i)


    def removeButton(self, index):
        ui.button(icon='remove', on_click = lambda e: self.removeInput(index)).props('dense')


    def addInput(self, index:int):
        newvalue = self.input_node_generator()
        self.params["list"].value.insert(index, newvalue)
        self.ui.refresh()


    def removeInput(self, index:int):
        self.params["list"].value.pop(index)
        self.ui.refresh()


    def process(self, list:List[Any]) -> List[Any]:
        return list
    
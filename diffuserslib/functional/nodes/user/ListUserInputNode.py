from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from .UserInputNode import UserInputNode
from nicegui import ui

class ListUserInputNode(UserInputNode):
    """ user interface node to supply a variable size list of items """
    def __init__(self, type:type, name:str="list_user_input"):
        self.type = type
        self.value = []
        super().__init__(name)


    @ui.refreshable
    def ui(self):
        ui.button('Add', on_click = lambda e: self.add(len(self.value)))
        for i, item in enumerate(self.value):
            with ui.row():
                ui.label(f"Item {i+1}").classes('mr-2')
                self.removeButton(i)
                
    def removeButton(self, index):
        ui.button('Remove', on_click = lambda e: self.remove(index))


    def add(self, index:int):
        nevalue = self.type()
        self.value.append(nevalue)
        self.ui.refresh()


    def remove(self, index:int):
        self.value.pop(index)
        self.ui.refresh()


    def process(self) -> List[Any]:
        return self.value
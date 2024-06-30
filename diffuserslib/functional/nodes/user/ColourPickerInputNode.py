from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from nicegui import ui
from diffuserslib.functional.types import *
from typing import Dict


class ColourPickerInputNode(UserInputNode):
    def __init__(self, mandatory:bool=False, name:str="colour_palette_user_input"):
        self.colour = "#000000"
        self.mandatory = mandatory
        self.enabled = True
        self.button = None
        super().__init__(name)


    def getValue(self):
        return self.colour
    

    def setValue(self, value):
        self.colour = value


    @ui.refreshable
    def gui(self):
        with ui.column():
            ui.label(f"{self.node_name}")
            with ui.row():
                if(not self.mandatory):
                    ui.checkbox(value=self.enabled, on_change=lambda e: self.setEnabled(e))
                with ui.button(icon='colorize').style(f'background-color:{self.colour}!important') as pick_colour:
                    self.button = pick_colour
                    ui.color_picker(on_pick=lambda e: self.pickColour(e))


    def setEnabled(self, e):
        print(e)
        self.enabled = e.value


    def pickColour(self, e):
        assert self.button is not None
        self.colour = e.color
        self.button.style(f'background-color:{e.color}!important')
    

    def __deepcopy__(self, memo):
        new_node = ColourPickerInputNode(self.mandatory, self.name)
        new_node.colour = self.colour
        new_node.enabled = self.enabled
        return new_node


    def process(self) -> ColourType|None:
        if(self.enabled):
            colour = tuple(int(self.colour[i:i+2], 16) for i in (1, 3, 5))
            return colour
        else:
            return None



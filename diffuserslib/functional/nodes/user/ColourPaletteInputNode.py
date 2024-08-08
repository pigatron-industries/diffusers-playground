from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from nicegui import ui
from diffuserslib.functional.types.ColourPalette import *
from typing import Dict


class ColourPaletteInputNode(UserInputNode):
    def __init__(self, mandatory:bool=False, name:str="colour_palette_user_input"):
        # TODO add multi select palette
        self.palette_type = "gradient"
        self.colours = ["#000000", "#ffffff"]
        self.buttons = []
        self.mandatory = mandatory
        self.enabled = True
        super().__init__(name)


    def getValue(self):
        return self.colours
    

    def setValue(self, value):
        self.colours = value


    @ui.refreshable
    def gui(self):
        with ui.column():
            ui.label(f"{self.node_name}")
            with ui.row():
                if(not self.mandatory):
                    ui.checkbox(value=self.enabled, on_change=lambda e: self.setEnabled(e))
                self.buttons = []
                for i in range(len(self.colours)):
                    self.colourPicker(i)


    def colourPicker(self, i):
        with ui.button(icon='colorize').style(f'background-color:{self.colours[i]}!important') as pick_colour:
            self.buttons.append(pick_colour)
            ui.color_picker(on_pick=lambda e: self.pickColour(e, i))


    def setEnabled(self, e):
        print(e)
        self.enabled = e.value


    def pickColour(self, e, i):
        self.colours[i] = e.color
        self.buttons[i].style(f'background-color:{e.color}!important')
    

    def __deepcopy__(self, memo):
        new_node = ColourPaletteInputNode(self.mandatory, self.name)
        new_node.colours = self.colours
        new_node.palette_type = self.palette_type
        new_node.enabled = self.enabled
        return new_node


    def processValue(self) -> ColourPalette|None:
        if(self.enabled):
            colours = [tuple(int(colour[i:i+2], 16) for i in (1, 3, 5)) for colour in self.colours]
            value = ColourPalette.fromGradient(colours[0], colours[1], 256)
            return value
        else:
            return None



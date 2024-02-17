from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from .UserInputNode import UserInputNode
from diffuserslib.util import FileDialog
from nicegui import ui
from PIL import Image


class ImageSelectInputNode(UserInputNode):
    def __init__(self, name:str="int_user_input"):
        self.filenames = []
        super().__init__(name)

    def getValue(self) -> List[str]:
        return self.filenames
    
    def setValue(self, value:List[str]):
        if(value is None):
            self.filenames = []
        else:
            self.filenames = value

    @ui.refreshable
    def ui(self):
        dialog = FileDialog(self.fileSelected, ["png", "jpg", "jpeg"])
        with ui.row().style("padding-top: 1.4em;"):
            if(len(self.filenames) == 0):
                ui.label('Select an image')
            elif(len(self.filenames) == 1):
                ui.label(self.filenames[0])
            else:
                ui.label(f"{len(self.filenames)} images selected")
            ui.button(icon='folder', on_click=dialog.open).props('dense')


    def fileSelected(self, selected_files:List[str]):
        self.filenames = selected_files
        self.ui.refresh()

    
    def process(self) -> Image.Image:
        if(len(self.filenames) == 0):
            raise Exception("Image not selected")
        return Image.open(self.filenames[0])
    



    

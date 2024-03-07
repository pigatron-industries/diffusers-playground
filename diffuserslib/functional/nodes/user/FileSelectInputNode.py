from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .UserInputNode import UserInputNode
from diffuserslib.util import FileDialog
from nicegui import ui


class FileSelectInputNode(UserInputNode):
    """A node that allows the user to select a file or multiple files. The output is a list of strings, each string being the path to a file."""

    def __init__(self, name:str="files_input"):
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
    def gui(self):
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
        self.gui.refresh()

    
    def process(self) -> list[str]:
        return self.filenames
    



    

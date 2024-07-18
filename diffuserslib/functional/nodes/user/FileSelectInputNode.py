from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .UserInputNode import UserInputNode
from diffuserslib.util import FileDialog
from nicegui import ui
from diffuserslib.interface.LocalFilePicker import LocalFilePicker
from diffuserslib.GlobalConfig import GlobalConfig


class FileSelectInputNode(UserInputNode):
    """A node that allows the user to select a file or multiple files. The output is a list of strings, each string being the path to a file."""

    extensions = {
        "image": ["png", "jpg", "jpeg"],
        "audio": ["wav", "mp3", "flac"],
        "video": ["mp4", "avi", "mkv"],
    }

    def __init__(self, filetype = "image", name:str="files_input"):
        self.filenames = []
        self.filetype = filetype
        super().__init__(name)

    def getValue(self) -> List[str]:
        return self.filenames
    
    def setValue(self, value:List[str]|None):
        if(value is None):
            self.filenames = []
        else:
            self.filenames = value

    @ui.refreshable
    def gui(self):
        self.dialog = LocalFilePicker(GlobalConfig.inputs_dirs[0], drives=GlobalConfig.inputs_dirs, multiple=True)  # TODO limit to extensions
        with ui.row().style("padding-top: 1.4em;"):
            if(len(self.filenames) == 0):
                ui.label(f'Select {self.filetype} file')
            elif(len(self.filenames) == 1):
                ui.label(self.filenames[0])
            else:
                ui.label(f"{len(self.filenames)} files selected")
            ui.button(icon='folder', on_click=self.selectFiles).props('dense')


    async def selectFiles(self):
        result = await self.dialog
        if(result is not None and len(result) > 0):
            self.filenames = result
            self.gui.refresh()


    def __deepcopy__(self, memo):
        new_node = FileSelectInputNode(filetype=self.filetype, name=self.node_name)
        new_node.filenames = self.filenames
        return new_node

    
    def process(self) -> list[str]:
        return self.filenames
    



    

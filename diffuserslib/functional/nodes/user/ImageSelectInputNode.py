from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .UserInputNode import UserInputNode
from diffuserslib.util import FileDialog
from nicegui import ui, events
from PIL import Image


class ImageSelectInputNode(UserInputNode):
    """A node that allows the user to uplaod a single image. The output is an image."""

    def __init__(self, name:str="image_input"):
        self.filename = None
        self.image = None
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.filename
    
    def setValue(self, value:str):
        if(value is None):
            self.filename = None
            self.image = None
        else:
            self.filename = value

    @ui.refreshable
    def ui(self):
        with ui.dialog() as dialog:
            ui.upload(on_upload=self.handleUpload)
        with ui.row().style("padding-top: 1.4em;"):
            if(self.image is None):
                ui.label('Select image')
            else:
                ui.image(self.image).style(f"max-width:128px; min-width:128px;")
            ui.button(icon='folder', on_click=dialog.open).props('dense')

    
    def handleUpload(self, e: events.UploadEventArguments):
        self.image = Image.open(e.content)
        self.ui.refresh()

    
    def process(self) -> Image.Image:
        if(self.image is None):
            raise Exception("Image not selected")
        return self.image

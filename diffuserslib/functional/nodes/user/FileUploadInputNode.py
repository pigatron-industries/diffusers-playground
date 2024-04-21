from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.interface.Clipboard import Clipboard
from .UserInputNode import UserInputNode
from nicegui import ui, events, run
from PIL import Image



class FileUploadInputNode(UserInputNode):
    """A node that allows the user to upload a single file. Subclass this to handle different file types."""

    def __init__(self, name:str="file_input"):
        self.filename = None
        self.content = None
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.filename
    
    def setValue(self, value:str):
        if(value is None):
            self.filename = None
            self.content = None
        else:
            self.filename = value

    @ui.refreshable
    def gui(self):
        with ui.dialog() as dialog:
            ui.upload(on_upload=self.handleUpload)
        with ui.row().style("padding-top: 1.4em;"):
            if(self.content is None):
                ui.label('Select file')
            else:
                self.previewContent()
            ui.button(icon='folder', on_click=dialog.open).props('dense')
            ui.button(icon='content_paste', on_click=lambda: run.io_bound(self.paste)).props('dense')


    def previewContent(self):
        pass


    def paste(self):
        raise NotImplementedError("Clipboard paste not implemented")

    
    def handleUpload(self, e: events.UploadEventArguments):
        raise NotImplementedError("File upload not implemented")

    
    def process(self) -> Image.Image:
        if(self.content is None):
            raise Exception("File not selected")
        return self.content



class ImageUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a single image. The output is an image."""

    def handleUpload(self, e: events.UploadEventArguments):
        self.content = Image.open(e.content)
        self.gui.refresh()

    def previewContent(self):
        if(self.content is None):
            return None
        with ui.column() as container:
            ui.image(self.content).style(f"max-width:128px; min-width:128px;")
            ui.label(f'{self.content.width} x {self.content.height} pixels')
        return container
    
    def paste(self):
        content = Clipboard.readObject()
        if(content is not None):
            self.content = content
        self.gui.refresh()
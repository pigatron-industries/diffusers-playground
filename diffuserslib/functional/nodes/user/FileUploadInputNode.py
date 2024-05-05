from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .UserInputNode import UserInputNode
from nicegui import ui, events, run
from PIL import Image



class FileUploadInputNode(UserInputNode):
    """A node that allows the user to upload a single file. Subclass this to handle different file types."""

    def __init__(self, mandatory:bool = True, display:str = "Select File", name:str="file_input"):
        self.filename = None
        self.content = None
        self.mandatory = mandatory
        self.display = display
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
                ui.label(self.display)
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

    
    def process(self) -> Any|None:
        if(self.content is None and self.mandatory):
            raise Exception("File not selected")
        return self.content

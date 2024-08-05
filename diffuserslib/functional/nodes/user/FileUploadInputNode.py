from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .UserInputNode import UserInputNode
from nicegui import ui, events, run
from PIL import Image



class FileUploadInputNode(UserInputNode):
    """A node that allows the user to upload a single file. Subclass this to handle different file types."""

    def __init__(self, mandatory:bool = True, multiple:bool=False, display:str = "Select file", name:str="file_input"):
        self.filename = None
        self.content = []
        self.mandatory = mandatory
        self.display = display
        self.multiple = multiple
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.filename
    
    def setValue(self, value:str|None):
        if(value is None):
            self.filename = None
            self.content = []
        else:
            self.filename = value

    @ui.refreshable
    def gui(self):
        with ui.dialog() as dialog:
            ui.upload(on_upload=self.handleUpload, on_multi_upload=self.handleMultiUpload, multiple=self.multiple)
        with ui.row().style("padding-top: 1.4em;"):
            with ui.column():
                ui.label(self.display)
                if(len(self.content) > 0):
                    self.previewContent()
            ui.button(icon='folder', on_click=dialog.open).props('dense')
            ui.button(icon='content_paste', on_click=lambda: run.io_bound(self.paste)).props('dense')


    def previewContent(self):
        pass


    def paste(self):
        raise NotImplementedError("Clipboard paste not implemented")

    
    def handleUpload(self, e: events.UploadEventArguments):
        raise NotImplementedError("File upload not implemented")
    

    def handleMultiUpload(self, e: events.UploadEventArguments):
        pass

    
    def process(self) -> Any|None:
        if(len(self.content) == 0 and self.mandatory):
            raise Exception(f"File not selected: {self.node_name}")
        if(self.multiple):
            return self.content
        else:
            return self.content[0]

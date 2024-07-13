from .FileUploadInputNode import FileUploadInputNode
from .VideoUploadInputNode import VideoUploadInputNode
from nicegui import ui, events
from PIL import Image
from diffuserslib.interface.Clipboard import Clipboard
import tempfile



class ImageUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a single image. The output is an image."""

    def __init__(self, mandatory:bool = True, multiple:bool=True, display:str = "Select image file", name:str="image_input"):
        self.preview = None
        self.content_temp = []
        super().__init__(mandatory, multiple, display, name)


    def handleUpload(self, e: events.UploadEventArguments):
        if(e.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))):
            self.addContent(Image.open(e.content))
        elif (e.name.lower().endswith(('.mp4', '.avi', '.mov'))):
            with tempfile.NamedTemporaryFile(suffix = ".mp4", delete=True) as temp_file:
                temp_file.write(e.content.read())
                temp_file.seek(0)
                frames, _ = VideoUploadInputNode.loadVideoFrames(temp_file.name)
                self.addContent(frames[-1])
        else:
            raise Exception("Invalid file type")
        self.gui.refresh()


    def handleMultiUpload(self, e: events.UploadEventArguments):
        self.content = self.content_temp
        self.content_temp = []
        self.gui.refresh()


    def clearContent(self):
        self.content = []
        self.content_temp = []
        self.preview = None
        self.gui.refresh()


    def addContent(self, image:Image.Image):
        if(self.multiple):
            self.content_temp.append(image)
        else:
            self.content_temp = [image]
            self.content = [image]
        self.preview = self.content_temp[0].copy()
        self.preview.thumbnail((128, 128))
        self.gui.refresh()


    def previewContent(self):
        if(len(self.content) == 0 or self.preview is None):
            return None
        with ui.column() as container:
            ui.image(self.preview).style(f"max-width:128px; min-width:128px;")
            ui.label(f'{len(self.content)} images - {self.content[0].width} x {self.content[0].height} pixels')
        return container
    

    def paste(self):
        clip = Clipboard.read()
        if(clip is not None):
            self.clearContent()
            self.addContent(clip.content)
        self.gui.refresh()
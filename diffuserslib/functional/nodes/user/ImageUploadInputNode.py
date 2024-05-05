from .FileUploadInputNode import FileUploadInputNode
from .VideoUploadInputNode import VideoUploadInputNode
from nicegui import ui, events
from PIL import Image
from diffuserslib.interface.Clipboard import Clipboard
import tempfile



class ImageUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a single image. The output is an image."""

    def __init__(self, mandatory:bool = True, display:str = "Select File", name:str="image_input"):
        self.preview = None
        super().__init__(mandatory, display, name)

    def handleUpload(self, e: events.UploadEventArguments):
        if(e.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))):
            image = Image.open(e.content)
            self.setContent(image)
        elif (e.name.lower().endswith(('.mp4', '.avi', '.mov'))):
            with tempfile.NamedTemporaryFile(suffix = ".mp4", delete=True) as temp_file:
                temp_file.write(e.content.read())
                temp_file.seek(0)
                frames, _ = VideoUploadInputNode.loadVideoFrames(temp_file.name)
                self.setContent(frames[-1])
        else:
            raise Exception("Invalid file type")
        self.gui.refresh()


    def setContent(self, content:Image.Image):
        self.content = content
        self.preview = content.copy()
        self.preview.thumbnail((128, 128))
        self.gui.refresh()


    def previewContent(self):
        if(self.content is None or self.preview is None):
            return None
        with ui.column() as container:
            ui.image(self.preview).style(f"max-width:128px; min-width:128px;")
            ui.label(f'{self.content.width} x {self.content.height} pixels')
        return container
    
    def paste(self):
        clip = Clipboard.read()
        if(clip is not None):
            self.content = clip.content
            self.preview = clip.preview
        self.gui.refresh()
from .FileUploadInputNode import FileUploadInputNode
from nicegui import ui, events
from PIL import Image
import cv2
import tempfile


class VideoUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a single image. The output is an image."""

    def __init__(self, mandatory:bool = True, display:str = "Select video file", name:str="video_input"):
        self.content = None
        self.fps = None
        super().__init__(mandatory, display, name)
        

    def handleUpload(self, e: events.UploadEventArguments):
        self.content = []
        with tempfile.NamedTemporaryFile(suffix = ".mp4", delete=True) as temp_file:
            temp_file.write(e.content.read())
            temp_file.seek(0)
            self.content, self.fps = VideoUploadInputNode.loadVideoFrames(temp_file.name)
            self.gui.refresh()


    @staticmethod
    def loadVideoFrames(file_name:str):
        frames = []
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        cap.release()
        return frames, fps


    def previewContent(self):
        if(self.content is None):
            return
        ui.image(self.content[0]).style(f"max-width:128px; min-width:128px;")
        with ui.column():
            ui.label(f"frames: {len(self.content)}")
            ui.label(f"fps: {self.fps}")
from .FileUploadInputNode import FileUploadInputNode
from diffuserslib.functional.types.Video import Video
from nicegui import ui, events
from PIL import Image
import cv2
import tempfile


class VideoUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a video. The output is a video."""

    def __init__(self, mandatory:bool = True, display:str = "Select video file", name:str="video_input"):
        self.content:Video|None = None
        self.preview:Image.Image|None = None
        self.framecount = 0
        super().__init__(mandatory, display, name)
        

    def handleUpload(self, e: events.UploadEventArguments):
        temp_file = tempfile.NamedTemporaryFile(suffix = ".mp4", delete=True)
        temp_file.write(e.content.read())
        temp_file.seek(0)
        cap = cv2.VideoCapture(temp_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read video file")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.preview = Image.fromarray(img)
        self.framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.content = Video(frame_rate = fps, file = temp_file)
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
        if(self.preview is None or self.content is None):
            return
        ui.image(self.preview).style(f"max-width:128px; min-width:128px;")
        with ui.column():
            ui.label(f"frames: {self.framecount}")
            ui.label(f"fps: {self.content.frame_rate}")


    def __deepcopy__(self, memo):
        new_node = VideoUploadInputNode(self.mandatory, self.display, self.name)
        new_node.content = self.content
        new_node.preview = self.preview
        new_node.framecount = self.framecount
        return new_node
from .FileUploadInputNode import FileUploadInputNode
from nicegui import ui, events
from PIL import Image
import cv2
import tempfile


class VideoUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a single image. The output is an image."""


    def handleUpload(self, e: events.UploadEventArguments):
        self.content = []
        with tempfile.NamedTemporaryFile(suffix = ".mp4", delete=True) as temp_file:
            temp_file.write(e.content.read())
            temp_file.seek(0)
            temp_file_name = temp_file.name

            cap = cv2.VideoCapture(temp_file_name)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.content.append(Image.fromarray(img))
            cap.release()
            self.gui.refresh()


    def previewContent(self):
        if(self.content is None):
            return
        ui.image(self.content[0]).style(f"max-width:128px; min-width:128px;")
        with ui.column():
            ui.label(f"frames: {len(self.content)}")
            ui.label(f"fps: {self.fps}")
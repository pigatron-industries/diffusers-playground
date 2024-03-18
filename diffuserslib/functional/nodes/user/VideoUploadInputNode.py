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
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                self.content.append(Image.fromarray(frame))
            cap.release()
            self.gui.refresh()


    def previewContent(self):
        if(self.content is None):
            return None
        return ui.image(self.content[0]).style(f"max-width:128px; min-width:128px;")
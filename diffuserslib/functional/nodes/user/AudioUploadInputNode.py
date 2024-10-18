from .FileUploadInputNode import FileUploadInputNode
from nicegui import ui, events
from diffuserslib.interface.Clipboard import Clipboard
from diffuserslib.functional.types import Audio
import tempfile
import librosa


class AudioUploadInputNode(FileUploadInputNode):
    """A node that allows the user to upload a single audio file. The output is an Audio."""

    def __init__(self, mandatory:bool = True, multiple:bool = False, sample_rate:int|None = 22050, mono:bool = True,
                 display:str = "Select audio file", name:str="audio_input"):
        self.content_temp = []
        self.sample_rate = sample_rate
        self.mono = mono
        super().__init__(mandatory, multiple, display, name)


    def handleUpload(self, e: events.UploadEventArguments):
        if(e.name.lower().endswith(("wav"))):
            temp_file = tempfile.NamedTemporaryFile(suffix = ".wav", delete=True)
            temp_file.write(e.content.read())
            temp_file.seek(0)
            audio_array, sample_rate = librosa.load(temp_file, sr=self.sample_rate, mono=self.mono)
            self.addContent(Audio(audio_array = audio_array, sample_rate = sample_rate, file = temp_file))
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
        self.gui.refresh()


    def addContent(self, audio:Audio):
        if(self.multiple):
            self.content_temp.append(audio)
        else:
            self.content_temp = [audio]
            self.content = [audio]
        # TODO store length, sample rate and channels
        self.gui.refresh()


    def previewContent(self):
        if(len(self.content) == 0):
            return None
        with ui.column() as container:
            # TODO display length, sample rate and channels
            ui.label(f'{len(self.content)} audios - sample rate: {self.content[0].sample_rate}')
        return container
    

    def paste(self):
        clip = Clipboard.read()
        if(clip is not None):
            self.clearContent()
            self.addContent(clip.content)
            self.content = self.content_temp
            self.content_temp = []
        self.gui.refresh()

    def __deepcopy__(self, memo):
        new_node = AudioUploadInputNode(self.mandatory, self.multiple, self.sample_rate, self.mono, self.display, self.name)
        new_node.content = self.content
        return new_node
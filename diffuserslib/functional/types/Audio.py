from typing import List, Callable
from scipy.io.wavfile import write as write_wav
import tempfile
import numpy as np


class Audio:
    def __init__(self, audio_array:np.ndarray, sample_rate:float, file:tempfile._TemporaryFileWrapper|None = None):
        self.audio_array = audio_array
        self.sample_rate = sample_rate
        self.file = file
        
    def write(self):
        if(self.file is None):
            self.file = tempfile.NamedTemporaryFile(suffix = ".wav", delete = True)
            print(f"Writing audio to {self.file.name}")
            write_wav(self.file.name, self.sample_rate, self.audio_array)


    def getFilename(self) -> str|None:
        if(isinstance(self.file, tempfile._TemporaryFileWrapper)):
            return self.file.name
        return None


AudioFuncType = Audio | Callable[[], Audio]
AudiosFuncType = List[Audio] | Callable[[], List[Audio]]

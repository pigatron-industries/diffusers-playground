from typing import List
from scipy.io.wavfile import write as write_wav
import tempfile
import io


class Audio:
    def __init__(self, audio_array, sample_rate:float):
        self.audio_array = audio_array
        self.sample_rate = sample_rate
        self.file = None
        
    def write(self):
        if(self.file is None):
            self.file = tempfile.NamedTemporaryFile(suffix = ".wav", delete = True)
            print(f"Writing audio to {self.file.name}")
            write_wav(self.file.name, self.sample_rate, self.audio_array)
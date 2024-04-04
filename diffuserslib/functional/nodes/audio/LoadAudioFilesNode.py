from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *

import librosa


class LoadAudioFilesNode(FunctionalNode):

    def __init__(self, 
                 files:StringsFuncType,
                 sample_rate:float|None = None,
                 mono:bool = True,
                 name:str="load_audio"):
        super().__init__(name)
        self.addParam("files", files, list[str])
        self.sample_rate = sample_rate
        self.mono = mono
        
        
    def process(self, files:list[str]) -> list[Audio]:
        audios = []
        for file in files:
            audio_array, sample_rate = librosa.load(file, sr=self.sample_rate, mono=self.mono)
            audios.append(Audio(audio_array, sample_rate))
        return audios
    
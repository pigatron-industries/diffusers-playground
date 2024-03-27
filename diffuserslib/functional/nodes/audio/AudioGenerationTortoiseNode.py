from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *

from tortoise.api import TextToSpeech


class AudioGenerationTortoiseNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 name:str="audio_bark"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.tts = None
        
        
    def process(self, prompt:str):
        if (self.tts is None):
            self.tts = TextToSpeech()
        audio_array = self.tts.tts_with_preset(prompt, preset = 'fast')
        audio_array = audio_array.cpu().numpy().squeeze()
        return Audio(audio_array, 24000)
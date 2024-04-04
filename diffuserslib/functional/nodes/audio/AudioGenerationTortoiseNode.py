from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio


class AudioGenerationTortoiseNode(FunctionalNode):

    def __init__(self, 
                 samples:StringsFuncType,
                 prompt:StringFuncType = "",
                 name:str="audio_bark"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("samples", samples, list[str])
        self.tts = None
        
        
    def process(self, prompt:str, samples:list[str]):
        if (self.tts is None):
            self.tts = TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)
        reference_clips = [load_audio(p, 22050) for p in samples]
        audio_array = self.tts.tts_with_preset(prompt,voice_samples = reference_clips, preset = 'standard')
        audio_array = audio_array.cpu().numpy().squeeze()
        return Audio(audio_array, 24000)
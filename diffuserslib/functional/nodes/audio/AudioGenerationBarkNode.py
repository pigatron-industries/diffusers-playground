from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib import pilToCv2
from transformers import AutoProcessor, BarkModel


class AudioGenerationBarkNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 voice:StringFuncType = "v2/en_speaker_6",
                 name:str="audio_bark"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("voice", voice, str)
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        
        
    def process(self, prompt:str, voice:str):
        inputs = self.processor(prompt, voice_preset=voice)
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return Audio(audio_array, model.generation_config.sample_rate)
from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *

from tortoise.api import TextToSpeech

import librosa
import torch


class AudioGenerationTortoiseNode(FunctionalNode):

    def __init__(self, 
                 samples:AudiosFuncType,
                 prompt:StringFuncType = "",
                 name:str="ausio_tortoise"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("samples", samples, List[Audio])
        self.tts = None
        
        
    def process(self, prompt:str, samples:List[Audio]):
        if (self.tts is None):
            self.tts = TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)

        if (samples is not None and len(samples) > 0):
            reference_clips = [self.prepare_audio(sample) for sample in samples]
            conditioning_latents = self.tts.get_conditioning_latents(reference_clips)
        else:
            conditioning_latents = self.tts.get_random_conditioning_latents()
        
        audio_array = self.tts.tts(text = prompt, conditioning_latents = conditioning_latents)
        audio_array = audio_array.cpu().numpy().squeeze()
        return Audio(audio_array, 24000)
    

    def prepare_audio(self, audio:Audio):
        audio_array = audio.audio_array
        target_sample_rate = 22050
        if(audio.sample_rate != target_sample_rate):
            audio_array = librosa.resample(audio.audio_array, orig_sr = audio.sample_rate, target_sr = target_sample_rate)
        if(audio_array.ndim > 1):
            audio_array = audio_array.mean(axis=0)
        return torch.FloatTensor(audio_array).unsqueeze(0)
    
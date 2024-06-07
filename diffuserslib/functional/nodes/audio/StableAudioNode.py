from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.GlobalConfig import GlobalConfig

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from einops import rearrange
import torch
import huggingface_hub


class StableAudioNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 duration:FloatFuncType = 10.0,
                 name:str="stableaudio"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("duration", duration, float)
        self.model = None
        self.model_config = None
        
        
    def process(self, prompt:str, duration:float):
        if (self.model is None):
            self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            self.model.to(GlobalConfig.device)
        assert self.model is not None, "Model is not initialized"
        assert self.model_config is not None, "Model config is not initialized"
        sample_rate = self.model_config['sample_rate']
        sample_size = self.model_config['sample_size']

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration,
        }]

        output = generate_diffusion_cond(
            model=self.model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min = 0.3,
            sigma_max = 500,
            sampler_type = "dpmpp-3m-sde",
            device=GlobalConfig.device
        )

        output = rearrange(output, 'b d n -> d (b n)')
        audio_array = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).cpu().numpy()
        return Audio(audio_array, sample_rate)

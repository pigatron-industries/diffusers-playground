from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.GlobalConfig import GlobalConfig
from diffusers import StableAudioPipeline
import torch

MAX_SEED = 4294967295

class StableAudioNode(FunctionalNode):

    def __init__(self, 
                 prompt:StringFuncType = "",
                 negative_prompt:StringFuncType = "",
                 duration:FloatFuncType = 10.0,
                 steps:FloatFuncType = 100,
                 cfg_scale:IntFuncType = 7,
                 seed:IntFuncType = 0,
                 initaudio:AudioFuncType|None = None,
                 name:str="stableaudio"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("negative_prompt", negative_prompt, str)
        self.addParam("duration", duration, float)
        self.addParam("steps", steps, int)
        self.addParam("cfg_scale", cfg_scale, float)
        self.addParam("seed", seed, int)
        self.addParam("initaudio", initaudio, Audio)
        self.model = None
        
        
    def process(self, prompt:str, negative_prompt:str, duration:float, steps:int, cfg_scale:float, seed:int|None, initaudio:Audio|None):
        if (self.model is None):
            self.model = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(GlobalConfig.device)
        assert self.model is not None, "Model is not initialized"

        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator("cpu").manual_seed(seed)
        
        params = {}
        params['prompt'] = prompt
        params['negative_prompt'] = negative_prompt
        params['num_inference_steps'] = steps
        params['audio_end_in_s'] = duration
        params['guidance_scale'] = cfg_scale
        params['generator'] = generator
        if initaudio is not None:
            params["initial_audio_waveforms"] = torch.from_numpy(initaudio.audio_array)
            params["initial_audio_sampling_rate"] = initaudio.sample_rate
        
        audio = self.model(params).audios
        
        audio_array = audio[0].T.float().cpu().numpy()
        return Audio(audio_array, self.model.vae.sampling_rate)

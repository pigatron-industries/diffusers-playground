from diffuserslib.functional.types import *
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.GlobalConfig import GlobalConfig
from diffusers import StableAudioPipeline
import torch

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from einops import rearrange

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
                 noise_level:FloatFuncType = 1.0,
                 name:str="stableaudio"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("negative_prompt", negative_prompt, str)
        self.addParam("duration", duration, float)
        self.addParam("steps", steps, int)
        self.addParam("cfg_scale", cfg_scale, float)
        self.addParam("seed", seed, int)
        self.addParam("initaudio", initaudio, Audio)
        self.addParam("noise_level", noise_level, float)
        self.model = None
        
        
    def process(self, prompt:str, negative_prompt:str, duration:float, steps:int, cfg_scale:float, seed:int|None, initaudio:Audio|None, noise_level:float):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        # return self.diffusers(prompt=prompt, negative_prompt=negative_prompt, duration=duration, steps=steps, cfg_scale=cfg_scale, seed=seed, initaudio=initaudio)
        return self.stable_audio_tools(prompt=prompt, negative_prompt=negative_prompt, duration=duration, steps=steps, cfg_scale=cfg_scale, seed=seed, 
                                       initaudio=initaudio, noise_level=noise_level)


    def stable_audio_tools(self, prompt:str, negative_prompt:str, duration:float, steps:int, cfg_scale:float, seed:int, initaudio:Audio|None, noise_level:float):
        if (self.model is None):
            self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            self.model.to(GlobalConfig.device)
        assert self.model is not None, "Model is not initialized"
        assert self.model_config is not None, "Model config is not initialized"
        sample_rate = self.model_config['sample_rate']
        sample_size = self.model_config['sample_size']

        params = {}
        params['model'] = self.model
        params['seed'] = seed
        params['steps'] = steps
        params['cfg_scale'] = cfg_scale
        params['conditioning'] = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration,
        }]
        params['negative_conditioning'] = [{
            "prompt": negative_prompt,
            "seconds_start": 0,
            "seconds_total": duration,
        }]
        params['sample_size'] = sample_size
        params['sigma_min'] = 0.3
        params['sigma_max'] = 500
        params['sampler_type'] = "dpmpp-3m-sde"
        params['device'] = GlobalConfig.device
        if initaudio is not None:
            audio = torch.from_numpy(initaudio.audio_array).div(2)
            params["init_audio"] = (initaudio.sample_rate, audio)
            params["init_noise_level"] = noise_level
        output = generate_diffusion_cond(**params)

        output = rearrange(output, 'b d n -> d (b n)')
        audio_array = output.to(torch.float32).clamp(-1, 1).cpu().numpy()
        audio_array = np.transpose(audio_array, (1, 0))
        return Audio(audio_array, sample_rate)



    def diffusers(self, prompt:str, negative_prompt:str, duration:float, steps:int, cfg_scale:float, seed:int, initaudio:Audio|None):
        if (self.model is None):
            self.model = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(GlobalConfig.device)
            # fixes bug where sample_size is a float
            self.model.transformer.config.sample_size = int(self.model.transformer.config.sample_size)
        assert self.model is not None, "Model is not initialized"

        generator = torch.Generator("cpu").manual_seed(seed)
        
        params = {}
        params['prompt'] = prompt
        params['negative_prompt'] = negative_prompt
        params['num_inference_steps'] = steps
        params['audio_end_in_s'] = duration
        params['guidance_scale'] = cfg_scale
        params['generator'] = generator
        if initaudio is not None:
            params["initial_audio_waveforms"] = torch.from_numpy(np.expand_dims(initaudio.audio_array, axis=0)).to(GlobalConfig.device)
            params["initial_audio_sampling_rate"] = initaudio.sample_rate

        audio = self.model(**params).audios[0]
        
        audio_array = audio.T.float().cpu().numpy()
        return Audio(audio_array, self.model.vae.sampling_rate)
from ..DiffusersModelPresets import DiffusersModel
from ..StringUtils import mergeDicts
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
import torch

class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel, controlmodel:str = None):
        self.preset = preset
        self.controlmodel = controlmodel


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, preset:DiffusersModel, cls, device, safety_checker=True, controlmodel=None, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset, controlmodel)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        # pipeline.enable_model_cpu_offload()
        # pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)

    def createPipelineArgs(self, preset, **kwargs):
        args = {}
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.revision is not None):
            args['revision'] = preset.revision
            if(preset.revision == 'fp16'):
                args['torch_dtype'] = torch.float16
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        # if(self.cache_dir is not None):
        #     args['cache_dir'] = self.cache_dir
        return mergeDicts(args, kwargs)
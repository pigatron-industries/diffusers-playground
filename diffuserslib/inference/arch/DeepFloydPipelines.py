from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch


class DeepFloydPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, safety_checker=True, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(params.modelConfig, cls, **kwargs)
        super().__init__(params, inferencedevice)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        self.pipeline = cls.from_pretrained(preset.modelpath, **args)
        self.pipeline2 = IFSuperResolutionPipeline.from_pretrained(preset.data['id2'], text_encoder=None, **args)
        self.pipeline3 = DiffusionPipeline.from_pretrained(preset.data['id3'], **args)

    def createPipelineArgs(self, preset:DiffusersModel, **kwargs):
        args = {}
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.revision is not None):
            args['variant'] = preset.revision
            if(preset.revision == 'fp16'):
                args['torch_dtype'] = torch.float16
        return mergeDicts(args, kwargs)
    
    def diffusers_inference(self, prompt, negprompt, seed, width, height, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        # if(scheduler is not None):
        #     self.loadScheduler(scheduler)
        prompt_embeds, negprompt_embeds = self.pipeline.encode_prompt(prompt=prompt, negative_prompt=negprompt)
        ptimage = self.pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negprompt_embeds, width=int(width/16), height=(int(height/16)), generator=generator, output_type="pt", **kwargs).images
        image = self.pipeline2(image=ptimage, prompt_embeds=prompt_embeds, negative_prompt_embeds=negprompt_embeds, generator=generator, output_type="pil", **kwargs).images[0]
        image = self.pipeline3(image=image, prompt=prompt, negative_prompt=negprompt, generator=generator, output_type="pil", **kwargs).images[0]
        return image, seed


class DeepFloydTextToImagePipelineWrapper(DeepFloydPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        super().__init__(cls=IFPipeline, params=params, device=device)

    def inference(self, params:GenerationParameters):
        return super().diffusers_inference(prompt=params.prompt, negprompt=params.negprompt, seed=params.seed, guidance_scale=params.cfgscale, 
                                           num_inference_steps=params.steps, scheduler=params.scheduler, width=params.width, height=params.height)

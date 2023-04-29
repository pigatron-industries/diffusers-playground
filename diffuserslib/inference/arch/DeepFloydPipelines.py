from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch


class DeepFloydPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        self.inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        self.pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)
        self.pipeline2 = IFSuperResolutionPipeline.from_pretrained(preset.data['id2'], text_encoder=None, **args)

    def createPipelineArgs(self, preset, **kwargs):
        args = {}
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.revision is not None):
            args['variant'] = preset.revision
            if(preset.revision == 'fp16'):
                args['torch_dtype'] = torch.float16
        return mergeDicts(args, kwargs)
    
    def inference(self, prompt, negprompt, seed, width, height, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        # if(scheduler is not None):
        #     self.loadScheduler(scheduler)
        prompt_embeds, negprompt_embeds = self.pipeline.encode_prompt(prompt=prompt, negative_prompt=negprompt)
        ptimage = self.pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negprompt_embeds, width=int(width/4), height=(int(height/4)), generator=generator, output_type="pt", **kwargs).images
        image = self.pipeline2(image=ptimage, prompt_embeds=prompt_embeds, negative_prompt_embeds=negprompt_embeds, generator=generator, output_type="pil", **kwargs).images[0]
        return image, seed


class DeepFloydTextToImagePipelineWrapper(DeepFloydPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(IFPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, seed, scale, steps, scheduler, width, height, **kwargs):
        return super().inference(prompt=prompt, negprompt=negprompt, seed=seed, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler, width=width, height=height)

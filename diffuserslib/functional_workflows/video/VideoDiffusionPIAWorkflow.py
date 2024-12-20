from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoDiffusionPIAWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Personalized Image Animator", Video, workflow=True, subworkflow=True)

    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        models_input = DiffusionModelUserInputNode(basemodels=["sd_1_5"])
        size_input = SizeUserInputNode(value = (512, 512))
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        strength_input = FloatUserInputNode(value = 1.0, name = "strength")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")
        frames_input = IntUserInputNode(value = 16, name = "frames")

        resize_image = ResizeImageNode(image = image_input, size = size_input, type = ResizeImageNode.ResizeType.STRETCH)

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        animatediff = VideoDiffusionPIANode(image = resize_image,
                                            models = models_input, 
                                            size = size_input, 
                                            prompt = prompt_processor,
                                            negprompt = negprompt_input,
                                            seed = seed_input,
                                            steps = steps_input,
                                            strength = strength_input,
                                            cfgscale = cfgscale_input,
                                            scheduler = scheduler_input,
                                            frames = frames_input)
        frames_to_video = FramesToVideoNode(frames = animatediff, fps = 7.5)
        
        models_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))
        return frames_to_video
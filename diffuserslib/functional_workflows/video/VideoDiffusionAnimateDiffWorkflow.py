from diffuserslib.functional import *


class VideoDiffusionAnimateDiffWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion AnimateDiff", Video, workflow=True, subworkflow=True)

    def build(self):
        models_input = DiffusionModelUserInputNode(basemodels=["sd_1_5"])
        size_input = SizeUserInputNode(value = (512, 512))
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")
        frames_input = IntUserInputNode(value = 16, name = "frames")

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        animatediff = VideoDiffusionAnimateDiffNode(models = models_input, 
                                                        size = size_input, 
                                                        prompt = prompt_processor,
                                                        negprompt = negprompt_input,
                                                        seed = seed_input,
                                                        steps = steps_input,
                                                        cfgscale = cfgscale_input,
                                                        scheduler = scheduler_input,
                                                        frames = frames_input)
        frames_to_video = FramesToVideoNode(frames = animatediff, fps = 7.5)
        
        models_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))
        return frames_to_video
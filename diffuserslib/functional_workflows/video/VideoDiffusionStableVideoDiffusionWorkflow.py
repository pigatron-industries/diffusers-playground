from diffuserslib.functional import *


class VideoDiffusionStableVideoDiffussionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Stable Video Diffusion", Video, workflow=True, subworkflow=True)

    def build(self):
        model_input = DiffusionModelUserInputNode(basemodels=["svd_1_0"], name="model")
        size_input = SizeUserInputNode(value = (512, 512), name="size")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")
        frames_input = IntUserInputNode(value = 16, name = "frames")

        image_input = ImageSelectInputNode(name = "image")
        conditioning_inputs = [ConditioningInputNode(image = image_input, model = "initimage")]

        animatediff = VideoDiffusionStableVideoDiffusionNode(models = model_input, 
                                                            size = size_input, 
                                                            seed = seed_input,
                                                            steps = steps_input,
                                                            cfgscale = cfgscale_input,
                                                            scheduler = scheduler_input,
                                                            frames = frames_input,
                                                            conditioning_inputs = conditioning_inputs)
        frames_to_video = FramesToVideoNode(frames = animatediff, fps = 7.5)
        
        # model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))
        return frames_to_video
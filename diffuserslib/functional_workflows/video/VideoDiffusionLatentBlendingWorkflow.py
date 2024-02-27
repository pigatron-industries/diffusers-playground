from diffuserslib.functional import *
from diffuserslib.functional_workflows.ImageDiffusionConditioningWorkflow import ImageDiffusionConditioningWorkflow


class VideoDiffusionLatentBlendingWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion Latent Blending", Image.Image, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        models_input = DiffusionModelUserInputNode()
        size_input = SizeUserInputNode(value = (512, 512))
        prompt1_input = TextAreaInputNode(value = "", name="prompt1")
        prompt2_input = TextAreaInputNode(value = "", name="prompt2")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed1_input = SeedUserInputNode(value = None, name="seed1")
        seed2_input = SeedUserInputNode(value = None, name="seed2")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")

        latent_blending = LatentBlendingNode(models = models_input, 
                                             loras = [],
                                             size = size_input,
                                             prompt1 = prompt1_input,
                                             prompt2 = prompt2_input,
                                             negprompt = negprompt_input,
                                             steps = steps_input,
                                             cfgscale = cfgscale_input,
                                             seed1 = seed1_input,
                                             seed2 = seed2_input,
                                             scheduler = scheduler_input)

        # num_frames_input = IntUserInputNode(value = 20, name = "num_frames")
        # frame_aggregator = FrameAggregatorNode(frame = diffusion, num_frames = num_frames_input)
        # frames_to_video = FramesToVideoNode(frames = frame_aggregator, fps = 10)
        return latent_blending

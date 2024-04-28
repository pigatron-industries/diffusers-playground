from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoDiffusionLatentBlendingWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Latent Blending", Video, workflow=True, subworkflow=False, realtime=False)


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
        max_branches_input = IntUserInputNode(value = 10, name = "max_branches")
        depth_strength_input = FloatUserInputNode(value = 0.5, name = "depth_strength")
        total_frames_input = IntUserInputNode(value = 60, name = "total_frames")
        fps_input = FloatUserInputNode(value = 7.5, name = "fps")

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
                                             scheduler = scheduler_input,
                                             max_branches=max_branches_input,
                                             depth_strength=depth_strength_input)
        
        linear_interpolation = FramesLinearInterpolationNode(frames = latent_blending, total_frames = total_frames_input)
        frames_to_video = FramesToVideoNode(frames = linear_interpolation, fps = fps_input)
        return frames_to_video

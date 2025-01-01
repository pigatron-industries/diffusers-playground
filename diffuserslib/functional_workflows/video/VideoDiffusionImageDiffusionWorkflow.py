from diffuserslib.functional_workflows.image.ImageDiffusionConditioningWorkflow import ImageDiffusionConditioningWorkflow
from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoDiffusionImageDiffusionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Animated Image Diffusion", Video, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        fps_input = FloatUserInputNode(name = "fps", value = 30)
        num_frames_input = IntUserInputNode(value = 20, name = "num_frames")

        diffusion, feedback = ImageDiffusionConditioningWorkflow().build()
        frame_aggregator = FrameAggregatorNode(frame = diffusion, num_frames = num_frames_input)
        frames_to_video = FramesToVideoNode(frames = frame_aggregator, fps = fps_input)
        return frames_to_video, feedback

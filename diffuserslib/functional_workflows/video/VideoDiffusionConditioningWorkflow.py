from diffuserslib.functional import *
from diffuserslib.functional_workflows.ImageDiffusionConditioningWorkflow import ImageDiffusionConditioningWorkflow


class VideoDiffusionConditioningWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Diffusion - Animated Conditioning", Video, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        diffusion, feedback = ImageDiffusionConditioningWorkflow().build()

        num_frames_input = IntUserInputNode(value = 20, name = "num_frames")
        frame_aggregator = FrameAggregatorNode(frame = diffusion, num_frames = num_frames_input)
        frames_to_video = FramesToVideoNode(frames = frame_aggregator, fps = 10)
        return frames_to_video, feedback

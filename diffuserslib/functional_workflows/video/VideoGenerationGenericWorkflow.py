from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoGenerationGenericWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Generation - Generic", Video, workflow=True, subworkflow=False)


    def build(self):
        fps_input = FloatUserInputNode(name = "fps", value = 30)
        num_frames_input = IntUserInputNode(name = "num_frames", value = 20)
        frame_input = ImageUploadInputNode(name = "frame_input")

        frame_aggregator = FrameAggregatorNode(frame = frame_input, num_frames = num_frames_input)
        frames_to_video = FramesToVideoNode(frames = frame_aggregator, fps = fps_input)
        return frames_to_video
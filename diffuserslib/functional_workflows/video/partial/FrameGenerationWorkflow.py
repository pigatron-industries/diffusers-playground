from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoGenerationGenericWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Frame Generation", List[Image.Image], workflow=False, subworkflow=True)


    def build(self):
        num_frames_input = IntUserInputNode(name = "num_frames", value = 20)
        frame_input = ImageUploadInputNode(name = "frame_input")

        frame_aggregator = FrameAggregatorNode(frame = frame_input, num_frames = num_frames_input)
        return frame_aggregator
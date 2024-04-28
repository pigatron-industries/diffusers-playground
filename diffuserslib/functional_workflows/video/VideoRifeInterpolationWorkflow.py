from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoRifeInterpolationWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video - Rife Interpolation", Video, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        frames_input = VideoUploadInputNode()
        mult_input = IntUserInputNode(name = "frame_multiplier", value = 4)
        fps_input = FloatUserInputNode(name = "fps", value = 30)

        rife = FramesRifeInterpolationNode(frames = frames_input, multiply = mult_input)
        frames_to_video = FramesToVideoNode(frames = rife, fps = fps_input)
        return frames_to_video

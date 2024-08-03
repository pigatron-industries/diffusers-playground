from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video import *
from diffuserslib.functional.nodes.user import *


class VideoReverseWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video - Reverse", Video, workflow=True, subworkflow=True, realtime=False)


    def build(self):
        video_input = VideoUploadInputNode()
        fps_input = FloatUserInputNode(name = "fps", value = 30)

        reverse = VideoReverseNode(video = video_input)
        frames_to_video = FramesToVideoNode(frames = reverse, fps = fps_input)
        return frames_to_video

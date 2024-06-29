from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.animated.VideoFrameSplitterNode import *
from diffuserslib.functional.nodes.user.VideoUploadInputNode import *
from diffuserslib.functional.nodes.user.UserInputNode import *


class FrameSplitterWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image - Frame Splitter", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        video_input = VideoUploadInputNode(name = "video")
        start_frame = IntUserInputNode(value = 0, name = "start_frame")
        skip_frames = IntUserInputNode(value = 0, name = "skip_frames")
        frame = VideoFrameSplitterNode(video = video_input, start_frame = start_frame, skip_frames = skip_frames)
        return frame
    
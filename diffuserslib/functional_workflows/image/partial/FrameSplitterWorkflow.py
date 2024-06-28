from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.animated.VideoFrameSplitterNode import *
from diffuserslib.functional.nodes.user.VideoUploadInputNode import *


class FrameSplitterWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image - Frame Splitter", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        video_input = VideoUploadInputNode(name = "video")
        frame = VideoFrameSplitterNode(video = video_input)
        return frame
    
from diffuserslib.functional import *


class FrameSplitterWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image - Frame Splitter", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        video_input = VideoUploadInputNode(name = "video")
        frame = FrameSplitterNode(frames = video_input)
        return frame
    
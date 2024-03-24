from diffuserslib.functional import *


class VideoRifeInterpolationWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video - Rife Interpolation", Image.Image, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        frames_input = VideoUploadInputNode()
        mult_input = IntUserInputNode(name = "frame_multiplier", value = 4)
        fps_input = FloatUserInputNode(name = "fps", value = 30)

        rife = FramesRifeInterpolationNode(frames = frames_input, multiply = mult_input)
        frames_to_video = FramesToVideoNode(frames = rife, fps = fps_input)
        return frames_to_video

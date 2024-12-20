from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types import Video, FramesFuncType, FloatFuncType
from PIL import Image

import tempfile
import cv2
import numpy as np


class FramesToVideoNode(FunctionalNode):
    def __init__(self, 
                 frames:FramesFuncType,
                 fps:FloatFuncType = 7.5,
                 name:str = "frames_to_video"):
        super().__init__(name)
        self.addParam("frames", frames, List[Image.Image])
        self.addParam("fps", fps, float)


    def process(self, frames:List[Image.Image], fps:float) -> Video:
        temp_file = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = True)
        height = frames[0].height
        width = frames[0].width
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

        for frame in frames:
            if(frame is None):
                break
            np_array = np.array(frame)
            cv2_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
            out.write(cv2_image)

        out.release()
        print(temp_file.name)
        return Video(frames = frames, frame_rate = fps, file = temp_file)
    

    # TODO don't regenerate the video every time getProgress is called
    # def getProgress(self) -> WorkflowProgress|None:
    #     frames_input_node = self.params["frames"]
    #     fps = self.params["fps"]
    #     if (isinstance(frames_input_node, FunctionalNode) and isinstance(fps, float)):
    #         progress = frames_input_node.getProgress()
    #         if (progress is not None):
    #             video = self.process(progress.output, fps)
    #             return WorkflowProgress(progress.progress, video)
    #     return None
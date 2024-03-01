from re import T
from diffuserslib.batch.BatchRunner import evaluateArguments
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Vector, MovingBody, MovingBodiesFuncType

import tempfile
import cv2
import numpy as np



class FramesLinearInterpolationNode(FunctionalNode):
    def __init__(self, 
                 frames:FramesFuncType,
                 total_frames:IntFuncType,
                 name:str = "frames_linear_interpolation"):
        super().__init__(name)
        self.addParam("frames", frames, List[Image.Image])
        self.addParam("total_frames", total_frames, int)


    def process(self, frames:List[Image.Image], total_frames:int) -> List[Image.Image]:
        num_frames_add = total_frames - len(frames)
        interval = num_frames_add // (len(frames)-1)
        remainder = num_frames_add % (len(frames)-1)
        new_frames = [frames[0]]
        for i in range(1, len(frames)):
            gap = interval
            if remainder > 0:  #evenly distribute remainders
                gap += 1
                remainder -= 1
            for j in range(gap):
                t = (j+1) / (gap + 1)
                new_frame = Image.blend(frames[i-1], frames[i], t)
                new_frames.append(new_frame)
            new_frames.append(frames[i])
        return new_frames
    


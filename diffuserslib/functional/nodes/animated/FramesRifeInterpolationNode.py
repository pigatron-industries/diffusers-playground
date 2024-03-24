from re import T
from diffuserslib.batch.BatchRunner import evaluateArguments
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Vector, MovingBody, MovingBodiesFuncType

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models.rife.v4_15.RIFE_HDv3 import Model


class FramesRifeInterpolationNode(FunctionalNode):
    def __init__(self, 
                 frames:FramesFuncType,
                 multiply:IntFuncType,
                 name:str = "frames_rife_interpolation"):
        super().__init__(name)
        self.addParam("frames", frames, List[Image.Image])
        self.addParam("multiply", multiply, int)
        self.device = "cpu"
        self.frames = []
        self.num_frames = 0


    def process(self, frames:List[Image.Image], multiply:int) -> List[Image.Image]:
        model = Model()
        model.load_model("models/rife/v4_15", -1)
        model.eval()
        model.device()

        self.frames = []
        self.num_frames = ((len(frames)-1) * multiply) + 1
        for i in range(len(frames) - 1):
            image1 = frames[i]
            image2 = frames[i + 1]
            self.frames.append(image1)
            # int_frames = [ Image.new("RGB", (512, 512), (0, 0, 0)) ]
            int_frames = self.interpolate(model, image1, image2, multiply)
            self.frames.extend(int_frames)
        self.frames.append(frames[-1])

        return self.frames
    

    def interpolate(self, model, image1:Image.Image, image2:Image.Image, mult:int) -> List[Image.Image]:
        img1 = np.array(image1)
        img2 = np.array(image2)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        img2 = (torch.tensor(img2.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)

        n, c, h, w = img1.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img1 = F.pad(img1, padding)
        img2 = F.pad(img2, padding)

        int_frames = []
        n = mult
        for i in range(n-1):
            ratio = (i + 1) * 1. / n
            int_frame = model.inference(img1, img2, ratio)
            int_frame = int_frame[0] * 255
            int_frame = int_frame.byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            int_frames.append(Image.fromarray(int_frame))

        return int_frames

        
    def getProgress(self) -> WorkflowProgress|None:
        if len(self.frames) == 0:
            return WorkflowProgress(0, None)
        else:
            return WorkflowProgress(float(len(self.frames)) / float(self.num_frames), self.frames)
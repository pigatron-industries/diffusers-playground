from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib import pilToCv2
import cv2
import numpy as np


class CannyEdgeDetectionNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 threshold_low:IntFuncType = 100,
                 threshold_high:IntFuncType = 200,
                 name:str="resize_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("threshold_low", threshold_low, int)
        self.addParam("threshold_high", threshold_high, int)
        
        
    def process(self, image:Image.Image, threshold_low:int, threshold_high:int) -> Image.Image:
        cv2image = pilToCv2(image)
        cv2image = cv2.Canny(cv2image, threshold_low, threshold_high)
        cv2image = cv2image[:, :, None]
        cv2image = np.concatenate([cv2image, cv2image, cv2image], axis=2)
        return Image.fromarray(cv2image)
    
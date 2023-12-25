
from ..ImageProcessor import ImageProcessor, ImageContext
from ....ImageUtils import cv2ToPil, pilToCv2
from typing import Dict, Any, List, Callable
import cv2
import numpy as np
    

class MovementProcessor(ImageProcessor):
    def __init__(self, args, frames = 0, interpolation:Callable[[float], float]|None = None):
        self.interpolation = interpolation
        self.frames = frames
        self.frame = 0
        super().__init__(args)
        self.calcTimings()

    def calcTimings(self):
        if(self.interpolation is not None):
            step_size = 1 / self.frames
            frame_positions = [i*step_size for i in range(self.frames+1)]
            self.frametimings = [self.interpolation(x) for x in frame_positions]
            self.frametimings_diff = [self.frametimings[i+1] - self.frametimings[i] for i in range(len(self.frametimings) - 1)]

    def getFrameTimeDiff(self):
        """ Git time difference between current frame and previous frame, as a fraction of the whole transform """
        if(self.frame > 0):
            return self.frametimings_diff[self.frame-1]
        else:
            return 0



class RotateMovementProcessor(MovementProcessor):
    def __init__(self, angle:float = 90, zoom:float = 1, frames = 0, interpolation:Callable[[float], float]|None = None):
        args = {
            "angle":angle,
            "zoom":zoom
        }
        super().__init__(args, frames, interpolation)
        

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getFullImage()
        angle = args["angle"]
        zoom = args["zoom"]

        frame_zoom = zoom ** self.getFrameTimeDiff()
        frame_angle = angle * self.getFrameTimeDiff()

        image = self.rotate(image, angle=frame_angle, zoom=frame_zoom)

        self.frame = self.frame + 1
        outputImage.setFullImage(image)
        return outputImage


    def rotate(self, image, angle=0, zoom=1, xcentre=0, ycentre=0):
        image = pilToCv2(image)
        height, width = image.shape[:2]
        xcentre = (width/2)*xcentre + (width/2)
        ycentre = (height/2)*ycentre + (height/2)
        rotation_matrix = cv2.getRotationMatrix2D((xcentre, ycentre), angle, zoom)
        output = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return cv2ToPil(output)
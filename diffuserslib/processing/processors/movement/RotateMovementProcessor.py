
from ..ImageProcessor import ImageContext
from ..FrameProcessor import FrameProcessor
from ....ImageUtils import cv2ToPil, pilToCv2
from typing import Dict, Any, List, Callable
import cv2


class RotateMovementProcessor(FrameProcessor):
    def __init__(self, angle:float = 90, zoom:float = 1, interpolation:Callable[[float], float]|None = None):
        args = {
            "angle":angle,
            "zoom":zoom
        }
        super().__init__(args, interpolation)
        

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
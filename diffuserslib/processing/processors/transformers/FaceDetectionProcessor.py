from ..ImageProcessor import ImageProcessor, ImageContext
from controlnet_aux import MediapipeFaceDetector
from typing import Dict, Any, List


class FaceDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.face_detector = MediapipeFaceDetector()
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.face_detector(inputImages[0].getViewportImage())
        outputImage.setViewportImage(image)
        return outputImage

from ..ImageProcessor import ImageProcessor, ImageContext
from typing import Dict, Any, List
    
    

class CropProcessor(ImageProcessor):
    def __init__(self, size = (512, 768), position = (0.5, 0.5)):
        args = {
            "size": size,
            "position": position
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getViewportImage()
        width = args["size"][0]
        height = args["size"][1]
        left = int((image.width - width) * args["position"][0])
        top = int((image.height - height) * args["position"][1])
        newimage = image.crop((left, top, left+width, top+height))
        outputImage.setViewportImage(newimage)
        return outputImage

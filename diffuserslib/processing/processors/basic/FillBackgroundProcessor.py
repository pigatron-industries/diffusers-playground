from PIL import Image
from ..ImageProcessor import ImageProcessor, ImageContext
from typing import Dict, Any, List


class FillBackgroundProcessor(ImageProcessor):
    def __init__(self, background="white"):
        args = {
            "background": background
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getFullImage().copy()
        background = Image.new("RGBA", size=image.size, color=args["background"])
        background.alpha_composite(image, (0, 0))
        outputImage.setFullImage(background)
        return outputImage

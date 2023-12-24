from ..ImageProcessor import ImageProcessor, ImageContext
from PIL import Image
from typing import Dict, Any, List

from IPython.display import display


class CompositeProcessor(ImageProcessor):
    """ 
    """
    def __init__(self):
        args = {
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image1 = inputImages[0].getFullImage().convert("RGB")
        image2 = inputImages[1].getFullImage().convert("RGB")

        display(image1)
        print(image1.size)
        display(image2)
        print(image2.size)

        image = Image.blend(image1, image2, 0.5)

        outputImage.setFullImage(image)
        return outputImage
from ..ImageProcessor import ImageProcessor, ImageContext
from typing import Dict, Any, List
from PIL import ImageDraw


class DrawCheckerboardProcessor(ImageProcessor):
    def __init__(self, size=(64, 64), fill="black", start="black"):
        args = {
            "size": size,
            "fill": fill,
            "start": start
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getViewportImage()
        draw = ImageDraw.Draw(image)
        square_width = args["size"][0]
        square_height = args["size"][1]
        for i in range(0, image.size[0], square_width):
            for j in range(0, image.size[1], square_height):
                if(args["start"] == "black"):
                    if (i//square_width + j//square_height) % 2 == 0:
                        draw.rectangle((i, j, i+square_width, j+square_height), fill=args["fill"])
                else:
                    if (i//square_width + j//square_height) % 2 == 1:
                        draw.rectangle((i, j, i+square_width, j+square_height), fill=args["fill"])
        outputImage.setViewportImage(image)
        return outputImage


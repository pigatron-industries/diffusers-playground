from ..ImageProcessor import ImageProcessor, ImageContext
from ....batch import RandomPositionArgument
from typing import Dict, Any, List

from PIL import ImageDraw


class DrawRegularShapeProcessor(ImageProcessor):
    def __init__(self, position=RandomPositionArgument(), size=64, sides=4, rotation=0, fill="black", outline=None):
        args = {
            "position": position,
            "size": size,
            "fill": fill,
            "outline": outline,
            "sides": sides,
            "rotation": rotation
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getFullImage()
        draw = ImageDraw.Draw(image)
        pos = (args["position"][0] + inputImages[0].offset[0], args["position"][1] + inputImages[0].offset[1])
        draw.regular_polygon(bounding_circle=(pos, args["size"]), n_sides=args["sides"], rotation=args["rotation"], fill=args["fill"], outline=args["outline"])
        outputImage.setFullImage(image)
        return outputImage

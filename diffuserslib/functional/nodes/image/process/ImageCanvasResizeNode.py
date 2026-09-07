from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image


class ImageCanvasResizeNode(FunctionalNode):
    """ Places the input image centred on a transparent canvas of the given final size.
        The area outside the original image is left transparent (alpha = 0) so that
        ImageAlphaToMaskNode can derive an inpaint mask for it. """

    def __init__(self,
                 image:ImageFuncType,
                 size:SizeFuncType = (1024, 1024),
                 name:str="canvas_resize"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("size", size, SizeType)


    def process(self, image:Image.Image, size:SizeType) -> Image.Image:
        final_w, final_h = int(size[0]), int(size[1])
        canvas = Image.new("RGBA", (final_w, final_h), (0, 0, 0, 0))
        source = image.convert("RGBA")
        x = max(0, (final_w - source.width) // 2)
        y = max(0, (final_h - source.height) // 2)
        canvas.paste(source, (x, y), source)
        return canvas

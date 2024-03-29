from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from enum import Enum


class ResizeImageNode(FunctionalNode):

    class ResizeType(Enum):
        STRETCH = "stretch"
        FIT = "fit"
        EXTEND = "extend"

    class HorizontalAlign(Enum):
        LEFT = "left"
        CENTRE = "centre"
        RIGHT = "right"

    class VerticalAlign(Enum):
        TOP = "top"
        CENTRE = "centre"
        BOTTOM = "bottom"


    ResizeTypeFuncType = ResizeType | Callable[[], ResizeType]
    HorizontalAlignFuncType = HorizontalAlign | Callable[[], HorizontalAlign]
    VerticalAlignFuncType = VerticalAlign | Callable[[], VerticalAlign]

    def __init__(self, 
                 image:ImageFuncType, 
                 size=SizeFuncType, 
                 type:ResizeTypeFuncType = ResizeType.STRETCH,
                 halign:HorizontalAlignFuncType = HorizontalAlign.CENTRE,
                 valign:VerticalAlignFuncType = VerticalAlign.CENTRE,
                 fill:ColourFuncType = "black",
                 name:str="resize_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("size", size, Tuple[int, int])
        self.addParam("type", type, self.ResizeType)
        self.addParam("halign", halign, str)
        self.addParam("valign", valign, str)
        self.addParam("fill", fill, str)
        
        
    def process(self, image:Image.Image, size:Tuple[int, int], type:ResizeType, halign:VerticalAlign, valign:HorizontalAlign, fill:str) -> Image.Image|None:
        if(image is None):
            return None
        if(type == self.ResizeType.STRETCH):
            newimage = image.resize(size, resample=Image.Resampling.LANCZOS)
        if(type == self.ResizeType.FIT):
            ratio = min(size[0]/image.width, size[1]/image.height)
            image = image.resize((int(image.width*ratio), int(image.height*ratio)), resample=Image.Resampling.LANCZOS)
        if(type in [self.ResizeType.EXTEND, self.ResizeType.FIT]):
            newimage = Image.new("RGBA", size=size, color=fill)
            if (halign == self.HorizontalAlign.LEFT):
                x = 0
            elif (halign == self.HorizontalAlign.RIGHT):
                x = newimage.width - image.width
            else:
                x = int((newimage.width - image.width)/2)
            if (valign == self.VerticalAlign.TOP):
                y = 0
            elif (valign == self.VerticalAlign.BOTTOM):
                y = newimage.height - image.height
            else:
                y = int((newimage.height - image.height)/2)
            newimage.paste(image, (x, y))
        return newimage

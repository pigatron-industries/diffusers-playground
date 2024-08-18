from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class ZoomImageNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 zoom:FloatFuncType = 0.0, 
                 fill_colour:ColourFuncType = "black",
                 name:str="resize_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("zoom", zoom, float)
        self.addParam("fill_colour", fill_colour, str)
        
        
    def process(self, image:Image.Image|None, zoom:float, fill_colour:ColourType) -> Image.Image|None:
        if(image is None):
            return None
        width, height = image.size
        zoom_width = int(width * zoom)
        zoom_height = int(height * zoom)
        resize_image = image.resize((zoom_width, zoom_height))
        new_image = Image.new("RGB", (width, height), fill_colour)
        new_image.paste(resize_image, ((width - zoom_width) // 2, (height - zoom_height) // 2))
        return new_image
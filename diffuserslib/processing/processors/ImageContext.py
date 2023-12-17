from typing import Tuple, Optional
from PIL import Image


class ImageContext():
    def __init__(self, size:Tuple[int, int]|None, oversize:int=256):
        self.oversize = oversize
        self.size = size
        self.offset = (oversize, oversize)
        self.filename = None
        if(size is not None):
            self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
            self.image = Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2))
        else:
            self.viewport = (oversize, oversize, oversize, oversize)
            self.image = None

    def setViewportImage(self, image:Image.Image):
        self.image = Image.new("RGBA", size=(image.width+self.oversize*2, image.height+self.oversize*2))
        self.image.paste(image, self.offset)
        if(hasattr(image, "filename")):
            self.filename = getattr(image, "filename")
        self.calcSize()

    def getViewportImage(self) -> Optional[Image.Image]:
        if(self.image is not None):
            image = self.image.crop(self.viewport)
            if(self.filename is not None):
                setattr(image, "filename", self.filename)
            return image

    def calcSize(self):
        if(self.image is not None):
            self.size = (self.image.width-self.oversize*2, self.image.height-self.oversize*2)
            self.viewport = (self.oversize, self.oversize, self.oversize+self.size[0], self.oversize+self.size[1])
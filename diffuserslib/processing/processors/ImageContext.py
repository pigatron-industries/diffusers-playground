from typing import Tuple, Self
from PIL import Image
from IPython.display import display


class ImageContext():
    def __init__(self, size:Tuple[int, int]|None, oversize:int=256):
        self.oversize = oversize
        self.size = size
        self.offset = (oversize, oversize)
        self.filename = None
        self.image = None
        if(size is not None):
            self.viewport = (oversize, oversize, oversize+size[0], oversize+size[1])
            self.image = Image.new("RGBA", size=(size[0]+oversize*2, size[1]+oversize*2))
        else:
            self.viewport = (oversize, oversize, oversize, oversize)


    @classmethod
    def copy(cls, imageContext:Self) -> Self:
        return cls(imageContext.size, imageContext.oversize)


    def getFullImage(self) -> Image.Image:
        if(self.image is not None):
            return self.image.copy()
        else:
            raise Exception("No image in context")
    

    def setFullImage(self, image:Image.Image):
        self.image = image.copy()
        self.calcSize()


    def getViewportImage(self) -> Image.Image:
        if(self.image is not None):
            image = self.image.crop(self.viewport)
            if(self.filename is not None):
                setattr(image, "filename", self.filename)
            return image
        else:
            raise Exception("No image in context")


    def setViewportImage(self, image:Image.Image):
        newImage = Image.new("RGBA", size=(image.width+self.oversize*2, image.height+self.oversize*2))
        newImage.paste(image, self.offset)
        self.image = newImage
        if(hasattr(image, "filename")):
            self.filename = getattr(image, "filename")
        self.calcSize()


    def calcSize(self):
        if(self.image is not None):
            self.size = (self.image.width-self.oversize*2, self.image.height-self.oversize*2)
            self.viewport = (self.oversize, self.oversize, self.oversize+self.size[0], self.oversize+self.size[1])
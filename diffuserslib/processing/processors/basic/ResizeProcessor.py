from PIL import Image
from ..ImageProcessor import ImageProcessor, ImageContext
from typing import Union, Callable, Dict, Any, List
    

class ResizeProcessor(ImageProcessor):
    def __init__(self, 
                 type:Union[str,Callable] = "stretch", 
                 size:Union[tuple[int, int],Callable] = (512, 768), 
                 halign:Union[str,Callable] = "centre", 
                 valign:Union[str,Callable] = "centre", 
                 fill:Union[str,Callable] = "black"):
        args = {
            "type": type,
            "size": size,
            "halign": halign,
            "valign": valign,
            "fill": fill
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getViewportImage()
        width = args["size"][0]
        height = args["size"][1]
        newimage = image

        if(args["type"] == "stretch"):
            newimage = image.resize(args["size"], resample=Image.Resampling.LANCZOS)

        if(args["type"] == "fit"):
            ratio = min(args["size"][0]/image.width, args["size"][1]/image.height)
            image = image.resize((int(image.width*ratio), int(image.height*ratio)), resample=Image.Resampling.LANCZOS)

        if(args["type"] in ("extend", "fit")):
            newimage = Image.new("RGBA", size=(width, height), color=args["fill"])
            if (args["halign"] == "left"):
                x = 0
            elif (args["halign"] == "right"):
                x = newimage.width - image.width
            else:
                x = int((newimage.width - image.width)/2)
            if (args["valign"] == "top"):
                y = 0
            elif (args["valign"] == "bottom"):
                y = newimage.height - image.height
            else:
                y = int((newimage.height - image.height)/2)
            newimage.paste(image, (x, y))

        outputImage.setViewportImage(newimage)
        return outputImage

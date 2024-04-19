from pydantic.dataclasses import dataclass
from typing import List, Any
from PIL import Image
from diffuserslib.ImageUtils import base64EncodeImage, base64DecodeImage


class ModelConfig:
    arbitrary_types_allowed = True


class ClipboardContentType:
    STRING = "str"
    IMAGE = "image"


@dataclass(config=ModelConfig)
class ClipboardContent:
    contenttype:str = ""
    content:str = ""


class Clipboard:
    content:List[ClipboardContent]

    @staticmethod
    def write(content:ClipboardContent):
        Clipboard.content = [content]


    @staticmethod
    def read():
        if (len(Clipboard.content) == 0):
            return None
        else:
            return Clipboard.content[0]


    @staticmethod
    def clear():
        Clipboard.content = []


    @staticmethod
    def writeObject(object:Any):
        if (isinstance(object, str)):
            Clipboard.write(ClipboardContent(contenttype=ClipboardContentType.STRING, content=object))
        elif (isinstance(object, Image.Image)):
            Clipboard.write(ClipboardContent(contenttype=ClipboardContentType.IMAGE, content=base64EncodeImage(object)))
        else:
            raise Exception("Unsupported object type")
        

    @staticmethod
    def readObject():
        content = Clipboard.read()
        if (content == None):
            return None
        if (content.contenttype == ClipboardContentType.STRING):
            return content.content
        elif (content.contenttype == ClipboardContentType.IMAGE):
            return base64DecodeImage(content.content)
        else:
            return None
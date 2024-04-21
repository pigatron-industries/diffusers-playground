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
    content:Any = None


@dataclass(config=ModelConfig)
class ClipboardContentDTO:
    contenttype:str = ""
    content:str = ""


class Clipboard:
    content:List[ClipboardContent] = []

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
            contenttype=ClipboardContentType.STRING
        elif (isinstance(object, Image.Image)):
            contenttype=ClipboardContentType.IMAGE
        else:
            raise Exception("Unsupported object type")
        Clipboard.write(ClipboardContent(contenttype=contenttype, content=object))
        

    @staticmethod
    def readObject():
        content = Clipboard.read()
        if (content is not None):
            return content.content


    @staticmethod
    def writeDTO(dto:ClipboardContentDTO):
        if(dto.contenttype == ClipboardContentType.IMAGE):
            content = base64DecodeImage(dto.content)
        elif(dto.contenttype == ClipboardContentType.STRING):
            content = dto.content
        else:
            raise Exception("Unsupported content type")
        Clipboard.write(ClipboardContent(contenttype=dto.contenttype, content=content))


    @staticmethod
    def readDTO():
        content = Clipboard.read()
        if (content is not None):
            if(content.contenttype == ClipboardContentType.IMAGE):
                dtocontent = base64EncodeImage(content.content)
            elif(content.contenttype == ClipboardContentType.STRING):
                dtocontent = content.content
            else:
                raise Exception("Unsupported content type")
            return ClipboardContentDTO(contenttype=content.contenttype, content=dtocontent)
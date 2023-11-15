import json
from krita import Krita # type: ignore
from .sd_parameters import *
from dataclasses import dataclass, asdict

@dataclass
class ConfigDialogParameters(GenerationParameters):
    modifiers:str = ""
    modelGeneration:str = ""
    modelPreprocess:str = ""
    modelUpscale:str = ""
    modelBase:str = ""
    

@dataclass
class SDConfig:
    "This is Stable Diffusion Plugin Main Configuration"     
    url = "http://localhost:5000"
    width=512
    height=512
    params:ConfigDialogParameters=ConfigDialogParameters()

    def __post_init__(self):
        self.load()

    def load(self):
        str=Krita.instance().readSetting("SDPlugin", "Config", None)
        if (not str): return
        self.unserialize(str)

    def save(self):
        str=self.serialize()
        Krita.instance().writeSetting("SDPlugin", "Config", str)

    def serialize(self):
        obj={
            "url":self.url,
            "width":self.width, 
            "height":self.height,
            "params":asdict(self.params)
        }
        print("saving config:")
        print(obj)
        return json.dumps(obj)
    
    def unserialize(self, str):
        obj=json.loads(str)
        # print("loading config:")
        # print(obj)
        self.url=obj.get("url","http://localhost:7860")
        self.type=obj.get("type","Colab")
        self.inpaint_mask_blur=obj.get("inpaint_mask_blur",4)
        self.inpaint_mask_content=obj.get("inpaint_mask_content","latent noise")
        self.width=obj.get("width",512)
        self.height=obj.get("height",512)
        if "params" in obj:
            self.params=ConfigDialogParameters.fromDict(obj["params"])
        else:
            self.params=ConfigDialogParameters()

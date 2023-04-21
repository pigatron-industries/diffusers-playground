import json

class SDConfig:
    "This is Stable Diffusion Plugin Main Configuration"     
    url = "http://localhost:5000"
    type="Colab"
    inpaint_mask_blur=4
    inpaint_mask_content="latent noise"     
    width=512
    height=512    
    dlgData={
        "action": "txt2img",
        "model": "runwayml/stable-diffusion-v1-5",
        "prompt": "",
        "instruct": "",
        "negprompt": "",
        "seed": "",
        "steps": 15,
        "steps_update": 50,
        "num": 2,
        "modifiers": "highly detailed\n",
        "scale": 7.5,
        "strength": .75,
        "scheduler":"DPMSolverMultistepScheduler",
        "upscale_amount": 2,
        "upscale_method": "all",
        "tile_method": "singlepass",
        "tile_width": 640,
        "tile_height": 640,
        "tile_overlap": 128,
        "tile_alignmentx": "tile_centred",
        "tile_alignmenty": "tile_centred",
        "process": "",
        "controlmodels": []
    }


    def serialize(self):
        obj={
            "url":self.url,
            "type":self.type,
            "inpaint_mask_blur":self.inpaint_mask_blur, 
            "inpaint_mask_content":self.inpaint_mask_content,
            "width":self.width, 
            "height":self.height,
            "type": self.type,
            "dlgData":self.dlgData
        }
        return json.dumps(obj)
    def unserialize(self,str):
        obj=json.loads(str)
        self.url=obj.get("url","http://localhost:7860")
        self.type=obj.get("type","Colab")
        self.dlgData=obj["dlgData"]
        self.inpaint_mask_blur=obj.get("inpaint_mask_blur",4)
        self.inpaint_mask_content=obj.get("inpaint_mask_content","latent noise")
        self.width=obj.get("width",512)
        self.height=obj.get("height",512)
    def save(self):
        str=self.serialize(self)
        Krita.instance().writeSetting ("SDPlugin", "Config", str)
    def load(self):
        str=Krita.instance().readSetting ("SDPlugin", "Config",None)
        if (not str): return
        self.unserialize(self,str)
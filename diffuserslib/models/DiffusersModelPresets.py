import yaml
from typing import Dict


AUTOENCODER_MODEL_1_5 = 'stabilityai/sd-vae-ft-mse'

class DiffusersBaseModel:
    sd_1_4 = 'sd_1_4'
    sd_1_5 = 'sd_1_5'
    sd_2_0 = 'sd_2_0'
    sd_2_1 = 'sd_2_1'


class DiffusersPipelineType:
    text2img = 'text2img'
    img2img = 'img2img'
    inpaint = 'inpaint'
    upscale = 'upscale'
    controlnet = 'controlnet'
    text2img_controlnet = 'text2img_controlnet'
    img2img_controlnet = 'img2img_controlnet'
    inpaint_controlnet = 'inpaint_controlnet'
 

class DiffusersModel:
    def __init__(self, modelid: str, base: str, pipelinetypes: Dict[str, str], revision: str = None, stylephrase: str = None, vae = None, autocast: bool = True, location: str = 'hf', modelpath: str = None):
        self.modelid = modelid
        self.base = base
        self.pipelinetypes = pipelinetypes
        self.revision = revision
        self.stylephrase = stylephrase
        self.vae = vae
        self.autocast = autocast
        self.location = location
        if(modelpath is None):
            self.modelpath = modelid
        else:
            self.modelpath = modelpath

    def toDict(self):
        return {
            'modelid': self.modelid, 
            'base': self.base,
            'stylephrase': self.stylephrase,
        }


class DiffusersModelList:
    def __init__(self):
        self.models = {}
        self.basemodels = {}

    def load_from_file(self, filepath: str):
        filedata = yaml.safe_load(open(filepath, "r"))
        for key in filedata:
            modeldata = filedata[key]
            for basedata in modeldata:
                self.addBaseModel(base = basedata['base'], pipelinetypes = basedata['pipelines'])
                for modeldata in basedata['models']:
                    # print(model)
                    if (modeldata.get('autocast') != 'false'):
                        autocast = True
                    else:
                        autocast = False
                    self.addModel(modelid=modeldata['id'], base=basedata['base'], revision=modeldata.get('revision'), 
                                    stylephrase=modeldata.get('phrase'), vae=modeldata.get('vae'), autocast=autocast)

    def addBaseModel(self, base: str, pipelinetypes: Dict[str, str]):
        if base not in self.basemodels:
            self.basemodels[base] = {}
        self.basemodels[base]['pipelines'] = pipelinetypes

    def addModel(self, modelid: str, base: str, revision: str=None, stylephrase:str=None, vae=None, autocast=True, location='hf', modelpath=None):
        self.models[modelid] = DiffusersModel(modelid=modelid, base=base, pipelinetypes=self.basemodels[base]['pipelines'], revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)

    def addModels(self, models):
        self.models.update(models.models)

    def getModel(self, modelid):
        if modelid in self.models:
            return self.models[modelid]
        else:
            return DiffusersModel(modelid, None, None, None, None)
        
    def getModelsByType(self, pipelinetype):
        matchingmodels = {}
        for modelid, model in self.models.items():
            if model.pipelinetypes is not None and pipelinetype in model.pipelinetypes:
                matchingmodels[modelid] = model
        return matchingmodels

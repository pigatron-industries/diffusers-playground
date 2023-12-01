from dataclasses import dataclass
import yaml
from typing import Dict, List, Union


AUTOENCODER_MODEL_1_5 = 'stabilityai/sd-vae-ft-mse'

class DiffusersBaseModelType:
    sd_1_4 = 'sd_1_4'
    sd_1_5 = 'sd_1_5'
    sd_2_0 = 'sd_2_0'
    sd_2_1 = 'sd_2_1'
    sdxl_1_0 = 'sdxl_1_0'
 

class DiffusersBaseModel:
    def __init__(self, pipelinetypes:Dict[str, str]):
        self.pipelinetypes = pipelinetypes

@dataclass
class DiffusersModel:
    modelid: Union[str, List[str]]
    base: str
    pipelinetypes: Dict[str, str]
    revision: str|None = None
    stylephrase: str|None = None
    vae: str|None = None
    autocast: bool = True
    location: str = 'hf'
    modelpath: str|None = None
    data:Dict|None = None

    def __post_init__(self):
        if(self.modelpath is None and isinstance(self.modelid, str)):
            self.modelpath = self.modelid
        else:
            self.modelpath = self.modelpath

    def toDict(self):
        return {
            'modelid': self.modelid, 
            'base': self.base,
            'stylephrase': self.stylephrase,
        }
    

@dataclass
class DiffusersConditioningModel:
    modelid: str
    type: str
    inputtype: str = "RGB"
    


class DiffusersModelList:
    def __init__(self):
        self.models:Dict[str, DiffusersModel] = {}
        self.basemodels:Dict[str, DiffusersBaseModel] = {}

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
                                    stylephrase=modeldata.get('phrase'), vae=modeldata.get('vae'), autocast=autocast, pipelinetypes = basedata['pipelines'].copy(), data=modeldata)

    def addBaseModel(self, base: str, pipelinetypes: Dict[str, str]):
        if base not in self.basemodels:
            self.basemodels[base] = DiffusersBaseModel(pipelinetypes)
        for pipelinetype in pipelinetypes:
            self.basemodels[base].pipelinetypes[pipelinetype] = pipelinetypes[pipelinetype]

    def addModel(self, modelid: str, base: str, revision: str|None=None, stylephrase:str|None=None, vae=None, autocast=True, location='hf', modelpath=None, 
                 pipelinetypes:Dict[str, str]|None=None, data=None):
        if pipelinetypes is None:
            pipelinetypes = self.basemodels[base].pipelinetypes
        self.models[modelid] = DiffusersModel(modelid=modelid, base=base, pipelinetypes=pipelinetypes, revision=revision, 
                                              stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath, data=data)

    def addModels(self, models):
        self.models.update(models.models)

    def getBaseModel(self, base):
        return self.basemodels[base]
    
    def getModel(self, modelid):
        if modelid in self.models:
            return self.models[modelid]
        else:
            return DiffusersModel(modelid, None, None, None, None)
        
    def getModelsByType(self, pipelinetype) -> Dict[str, DiffusersModel]:
        matchingmodels = {}
        for modelid, model in self.models.items():
            if model.pipelinetypes is not None and pipelinetype in model.pipelinetypes:
                matchingmodels[modelid] = model
        return matchingmodels

    def getModelsByTypeAndBase(self, pipelinetype, base) -> Dict[str, DiffusersModel]:
        matchingmodels = {}
        for modelid, model in self.models.items():
            if model.pipelinetypes is not None and pipelinetype in model.pipelinetypes and model.base == base:
                matchingmodels[modelid] = model
        return matchingmodels
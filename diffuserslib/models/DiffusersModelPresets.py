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

class DiffusersModelType:
    generate = 'generate'
    inpaint = 'inpaint'
    controlimage = 'controlimage'
    controlnet = 'controlnet'
    t2iadapter = 't2iadapter'
    ipadapter = 'ipadapter'
 

class DiffusersBaseModel:
    def __init__(self, pipelinetypes:Dict[str, str]):
        self.pipelinetypes = pipelinetypes

@dataclass
class DiffusersModel:
    modelid: str
    base: str
    modeltypes: List[str]
    modelpath: str
    pipelinetypes: Dict[str, str]
    revision: str|None = None
    stylephrase: str|None = None
    vae: str|None = None
    autocast: bool = True
    preprocess: str|None = None
    location: str = 'hf'
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
    

def diffusersmodel_representer(dumper: yaml.Dumper, data: DiffusersModel) -> yaml.Node:
    return dumper.represent_list(data.modelid)
yaml.add_representer(DiffusersModel, diffusersmodel_representer)
    

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
                if('base' in basedata):
                    basemodel = self.addBaseModel(base = basedata['base'], pipelinetypes = basedata['pipelines'] if 'pipelines' in basedata else {})
                    for modeldata in basedata['models']:
                        # print(model)
                        if (modeldata.get('autocast') != 'false'):
                            autocast = True
                        else:
                            autocast = False
                        self.addModel(modelid=modeldata['id'], base=basedata['base'], modeltype = key, revision=modeldata.get('revision'), 
                                    stylephrase=modeldata.get('phrase'), vae=modeldata.get('vae'), preprocess=modeldata.get('preprocess'), 
                                    location=modeldata.get('location', 'hf'), 
                                    autocast=autocast, pipelinetypes = basemodel.pipelinetypes, data=modeldata)

    def addBaseModel(self, base: str, pipelinetypes: Dict[str, str]):
        if base not in self.basemodels:
            self.basemodels[base] = DiffusersBaseModel(pipelinetypes)
        self.basemodels[base].pipelinetypes.update(pipelinetypes)
        return self.basemodels[base]

    def addModel(self, modelid: str, base: str, modeltype: str, revision: str|None=None, stylephrase:str|None=None, vae=None, 
                 preprocess:str|None=None, autocast=True, location='hf', modelpath=None, pipelinetypes:Dict[str, str]|None=None, data=None):
        if pipelinetypes is None:
            pipelinetypes = self.basemodels[base].pipelinetypes
        modeltypes = [modeltype]
        if modelid in self.models:
            modeltypes.extend(self.models[modelid].modeltypes)
        self.models[modelid] = DiffusersModel(modelid=modelid, base=base, modeltypes=modeltypes, pipelinetypes=pipelinetypes, revision=revision, 
                                              stylephrase=stylephrase, vae=vae, autocast=autocast, preprocess=preprocess, location=location, modelpath=modelpath, data=data)

    def addModels(self, models):
        self.models.update(models.models)

    def getBaseModel(self, base):
        return self.basemodels[base]
    
    def getModel(self, modelid) -> DiffusersModel:
        if modelid in self.models:
            return self.models[modelid]
        else:
            return DiffusersModel(modelid, None, [], None, None)
        
    def getModelsByType(self, modeltype) -> Dict[str, DiffusersModel]:
        if modeltype == 'controlimage':
            modeltypes = ['controlnet', 't2iadapter', 'ipadapter']
        else:
            modeltypes = [modeltype]
        matchingmodels = {}
        for modelid, model in self.models.items():
            if any(mt in model.modeltypes for mt in modeltypes):
                matchingmodels[modelid] = model
        return matchingmodels

    def getModelsByTypeAndBase(self, modeltype, base) -> Dict[str, DiffusersModel]:
        if modeltype == 'conditioning':
            modeltypes = ['controlnet', 't2iadapter', 'ipadapter']
        else:
            modeltypes = [modeltype]
        matchingmodels = {}
        for modelid, model in self.models.items():
            if any(mt in model.modeltypes for mt in modeltypes) and model.base == base:
                matchingmodels[modelid] = model
        return matchingmodels
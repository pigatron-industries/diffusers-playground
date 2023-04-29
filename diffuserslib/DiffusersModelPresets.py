import yaml


AUTOENCODER_MODEL_1_5 = 'stabilityai/sd-vae-ft-mse'

class DiffusersBaseModel:
    sd_1_4 = 'sd_1_4'
    sd_1_5 = 'sd_1_5'
    sd_2_1 = 'sd_2_1'
 

class DiffusersModel:
    def __init__(self, modelid: str, base: str, revision: str = None, stylephrase: str = None, vae = None, autocast: bool = True, location: str = 'hf', modelpath: str = None):
        self.modelid = modelid
        self.base = base
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

    @classmethod
    def from_file(cls, filepath, key):
        filedata = yaml.safe_load(open(filepath, "r"))
        modellist = DiffusersModelList()
        if(key in filedata):
            modeldata = filedata[key]
            for basedata in modeldata:
                for model in basedata['models']:
                    # print(model)
                    if (model.get('autocast') != 'false'):
                        autocast = True
                    else:
                        autocast = False
                    modellist.addModel(modelid=model['id'], base=basedata['base'], revision=model.get('revision'), 
                                    stylephrase=model.get('phrase'), vae=model.get('vae'), autocast=autocast)
        return modellist

    def addModel(self, modelid, base, revision=None, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        self.models[modelid] = DiffusersModel(modelid, base, revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)

    def addModels(self, models):
        self.models.update(models.models)

    def getModel(self, modelid):
        if modelid in self.models:
            return self.models[modelid]
        else:
            return DiffusersModel(modelid, None, None, None, None)


AUTOENCODER_MODEL_1_5 = 'stabilityai/sd-vae-ft-mse'

class DiffusersBaseModel:
    sd_1_5 = 'sd_1_5'
    sd_2_0_512 = 'sd_2_0_512'
    sd_2_0_768 = 'sd_2_0_768'
    sd_2_1_512 = 'sd_2_1_512'
    sd_2_1_768 = 'sd_2_1_768'
 

class DiffusersModel:
    def __init__(self, modelid, base, fp16=True, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        self.modelid = modelid
        self.base = base
        self.fp16 = fp16
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

    def addModel(self, modelid, base, fp16=True, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        self.models[modelid] = DiffusersModel(modelid, base, fp16=fp16, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)

    def addModels(self, models):
        self.models.update(models.models)

    def getModel(self, modelid):
        if modelid in self.models:
            return self.models[modelid]
        else:
            return DiffusersModel(modelid, None, False, None, None)



def getHuggingFacePresetsList():
    presets = DiffusersModelList()

    # 2.1 768
    #                model                                  base                           fp16   phrase            vae   autocast
    presets.addModel('stabilityai/stable-diffusion-2-1',    DiffusersBaseModel.sd_2_1_768, True,  None,             None, False)

    # 2.1 512
    #                model                                    base                           fp16   phrase            vae
    presets.addModel('stabilityai/stable-diffusion-2-1-base', DiffusersBaseModel.sd_2_1_512, True,  None,             None)

    # 2.0 768
    #                model                                  base                           fp16   phrase            vae
    presets.addModel('stabilityai/stable-diffusion-2',      DiffusersBaseModel.sd_2_0_768, True,  None,             None)
    presets.addModel('nitrosocke/redshift-diffusion-768',   DiffusersBaseModel.sd_2_0_768, False, 'redshift style', None)

    # 2.0 512
    #                model                                  base                           fp16   phrase            vae
    presets.addModel('stabilityai/stable-diffusion-2-base', DiffusersBaseModel.sd_2_0_512, True,  None,             None)
    presets.addModel('nitrosocke/Future-Diffusion',         DiffusersBaseModel.sd_2_0_512, False, 'future style',   None)

    # 1.5
    #                model                                            base                       fp16   phrase                vae
    presets.addModel('runwayml/stable-diffusion-v1-5',                DiffusersBaseModel.sd_1_5, True,  None,                 AUTOENCODER_MODEL_1_5)
    presets.addModel('hassanblend/HassanBlend1.5.1.2',                DiffusersBaseModel.sd_1_5, False, None,                 AUTOENCODER_MODEL_1_5)
    presets.addModel('Linaqruf/anything-v3.0',                        DiffusersBaseModel.sd_1_5, False, None,                 None)
    presets.addModel('nitrosocke/redshift-diffusion',                 DiffusersBaseModel.sd_1_5, False, 'redshift style',     AUTOENCODER_MODEL_1_5)
    presets.addModel('Fictiverse/Stable_Diffusion_Microscopic_model', DiffusersBaseModel.sd_1_5, False, 'microscopic',        AUTOENCODER_MODEL_1_5)
    presets.addModel('Fictiverse/Stable_Diffusion_PaperCut_Model',    DiffusersBaseModel.sd_1_5, True,  'papercut',           AUTOENCODER_MODEL_1_5)
    presets.addModel('BunnyViking/rachelwalkerstylewatercolour',      DiffusersBaseModel.sd_1_5, True,  'rachelwalker style', AUTOENCODER_MODEL_1_5)
    presets.addModel('plasmo/woolitize',                              DiffusersBaseModel.sd_1_5, True,  'woolitize',          AUTOENCODER_MODEL_1_5)
    presets.addModel('plasmo/food-crit',                              DiffusersBaseModel.sd_1_5, False, 'food_crit',          AUTOENCODER_MODEL_1_5)
    presets.addModel('Aybeeceedee/knollingcase',                      DiffusersBaseModel.sd_1_5, False, 'knollingcase',       AUTOENCODER_MODEL_1_5)
    presets.addModel('wavymulder/Analog-Diffusion',                   DiffusersBaseModel.sd_1_5, False, 'analog style',       AUTOENCODER_MODEL_1_5)
    presets.addModel('dreamlike-art/dreamlike-diffusion-1.0',         DiffusersBaseModel.sd_1_5, False, 'dreamlikeart',       AUTOENCODER_MODEL_1_5)
    presets.addModel('prompthero/openjourney',                        DiffusersBaseModel.sd_1_5, False, 'mdjrny-v4 style',    AUTOENCODER_MODEL_1_5)
    return presets
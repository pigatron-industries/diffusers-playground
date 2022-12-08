
AUTOENCODER_MODEL_1_5 = 'stabilityai/sd-vae-ft-mse'

class DiffusersBaseModel:
    sd_1_5 = 1
    sd_2_0_512 = 2
    sd_2_0_768 = 3


class DiffusersModelLocation:
    local = 1
    huggingface = 2


class DiffusersModel:
    def __init__(self, modelid, base, fp16=True, stylephrase=None, vae=None, location=DiffusersModelLocation.huggingface):
        self.modelid = modelid
        self.base = base
        self.fp16 = fp16
        self.stylephrase = stylephrase
        self.vae = vae
        self.location = location


class DiffusersModelList:
    def __init__(self):
        self.models = {}

    def addModel(self, modelid, base, fp16=True, stylephrase=None, vae=None, location=DiffusersModelLocation.huggingface):
        self.models[modelid] = DiffusersModel(modelid, base, fp16, stylephrase, vae, location)

    def addModels(self, models):
        self.models.update(models.models)

    def getModel(self, modelid):
        if modelid in self.models:
            return self.models[modelid]
        else:
            return DiffusersModel('modelid', None, False, None, None)



def getHuggingFacePresetsList():
    presets = DiffusersModelList()

    # 2.0 768
    #                model                                  base                           fp16   phrase            vae
    presets.addModel('stabilityai/stable-diffusion-2',      DiffusersBaseModel.sd_2_0_768, True,  None,             None)
    presets.addModel('nitrosocke/redshift-diffusion-768',   DiffusersBaseModel.sd_2_0_768, False, 'redshift style', None)

    # 2.0 512
    #                model                                  base                           fp16   phrase            vae
    presets.addModel('stabilityai/stable-diffusion-2-base', DiffusersBaseModel.sd_2_0_512, True,  None,             None)
    presets.addModel('nitrosocke/Future-Diffusion',         DiffusersBaseModel.sd_2_0_512, False, 'future style',   None)

    # 1.5
    #                model                                            base                       fp16   phrase            vae
    presets.addModel('runwayml/stable-diffusion-v1-5',                DiffusersBaseModel.sd_1_5, True,  None,                 AUTOENCODER_MODEL_1_5)
    presets.addModel('hassanblend/hassanblend1.4',                    DiffusersBaseModel.sd_1_5, False, None,                 AUTOENCODER_MODEL_1_5)
    presets.addModel('Linaqruf/anything-v3.0',                        DiffusersBaseModel.sd_1_5, False, None,                 None)
    presets.addModel('nitrosocke/redshift-diffusion',                 DiffusersBaseModel.sd_1_5, False, 'redshift style',     AUTOENCODER_MODEL_1_5)
    presets.addModel('Fictiverse/Stable_Diffusion_Microscopic_model', DiffusersBaseModel.sd_1_5, True,  'microscopic',        AUTOENCODER_MODEL_1_5)
    presets.addModel('Fictiverse/Stable_Diffusion_PaperCut_Model',    DiffusersBaseModel.sd_1_5, True,  'papercut',           AUTOENCODER_MODEL_1_5)
    presets.addModel('BunnyViking/rachelwalkerstylewatercolour',      DiffusersBaseModel.sd_1_5, True,  'rachelwalker style', AUTOENCODER_MODEL_1_5)
    presets.addModel('plasmo/woolitize',                              DiffusersBaseModel.sd_1_5, True,  'woolitize',          AUTOENCODER_MODEL_1_5)
    presets.addModel('plasmo/food-crit',                              DiffusersBaseModel.sd_1_5, True,  '',                   AUTOENCODER_MODEL_1_5)
    presets.addModel('Aybeeceedee/knollingcase',                      DiffusersBaseModel.sd_1_5, False, 'knollingcase',       AUTOENCODER_MODEL_1_5)
    return presets
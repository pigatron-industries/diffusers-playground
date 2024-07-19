from dataclasses import dataclass
import yaml


@dataclass
class OllamaModel:
    modelid:str


class OllamaModels:
    modelList = {}

    @staticmethod
    def loadPresetFile(filepath):
        filedata = yaml.safe_load(open(filepath, "r"))
        if('ollama' in filedata):
            for modeldata in filedata['ollama']:
                modelid = modeldata['id']
                OllamaModels.modelList[modelid] = OllamaModel(modelid)

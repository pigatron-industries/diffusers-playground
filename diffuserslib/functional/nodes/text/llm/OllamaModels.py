from dataclasses import dataclass
import requests


@dataclass
class OllamaModel:
    modelid:str


class OllamaModels:
    url = "http://localhost:11434"
    modelList = {}

    @staticmethod
    def loadLocalModels():
        url = OllamaModels.url + "/api/tags"
        response = requests.get(url)
        if(response.status_code == 200):
            data = response.json()
            for model in data['models']:
                modelid = model['name']
                OllamaModels.modelList[modelid] = OllamaModel(modelid)
        return OllamaModels.modelList
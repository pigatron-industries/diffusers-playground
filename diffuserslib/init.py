from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from typing import List
import yaml
import os

class GlobalConfig:
    outputs_dir = None


def initializeDiffusers(configs:List[str]=["config.yml"], modelconfigs:List[str]=["modelconfig.yml"]):
    safety_checker = True
    device = "cuda"
    models = "./models"
    for config in configs:
        if os.path.exists(config):
            configdata = yaml.safe_load(open(config, "r"))
            device = configdata["device"]
            safety_checker = configdata["safety"]
            models = configdata["folders"]["models"]
            GlobalConfig.outputs_dir = configdata["folders"]["outputs"]
            break

    DiffusersPipelines.pipelines = DiffusersPipelines(device=device, safety_checker=safety_checker, localmodelpath=models)
    for modelconfig in modelconfigs:
        if os.path.exists(modelconfig):
            DiffusersPipelines.pipelines.loadPresetFile(modelconfig)

    for config in configs:
        if os.path.exists(config):
            configdata = yaml.safe_load(open(config, "r"))
            embeddings = configdata["folders"]["embeddings"]
            loras = configdata["folders"]["loras"]
            DiffusersPipelines.pipelines.loadTextEmbeddings(embeddings)
            DiffusersPipelines.pipelines.loadLORAs(loras)



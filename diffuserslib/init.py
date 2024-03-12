from diffuserslib.inference import DiffusersPipelines
from diffuserslib.functional import WorkflowRunner
from diffuserslib.GlobalConfig import GlobalConfig
from diffuserslib.functional.nodes.diffusers.RandomPromptProcessorNode import RandomPromptProcessorNode
from typing import List
import yaml
import os
import numpy as np
from PIL import Image


def initializeDiffusers(configs:List[str]=["config.yml"], modelconfigs:List[str]=["modelconfig.yml"], promptmods:List[str]=[]):
    safety_checker = True
    device = "cuda"
    models = "./models"
    for config in configs:
        if os.path.exists(config):
            configdata = yaml.safe_load(open(config, "r"))
            device = configdata["device"]
            safety_checker = configdata["safety"]
            folders = configdata["folders"]
            if ("models" in folders):
                models = folders["models"]
            if ("outputs" in folders):
                outputs_dir = folders["outputs"]
                WorkflowRunner.workflowrunner = WorkflowRunner(output_dir=outputs_dir)
            if ("inputs" in folders):
                GlobalConfig.inputs_dirs.extend(folders["inputs"])

    DiffusersPipelines.pipelines = DiffusersPipelines(device=device, safety_checker=safety_checker, localmodelpath=models)
    for modelconfig in modelconfigs:
        if os.path.exists(modelconfig):
            DiffusersPipelines.pipelines.loadPresetFile(modelconfig)

    for config in configs:
        if os.path.exists(config):
            configdata = yaml.safe_load(open(config, "r"))
            embeddings = configdata["folders"]["embeddings"]
            loras = configdata["folders"]["loras"]
            for embeddingdir in embeddings:
                DiffusersPipelines.pipelines.loadTextEmbeddings(embeddingdir)
            for loradir in loras:
                DiffusersPipelines.pipelines.loadLORAs(loradir)

    for promptmod in promptmods:
        RandomPromptProcessorNode.loadModifierDictFile(promptmod)



# Initialise yaml representers for yml dumps
def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())
def image_representer(dumper: yaml.Dumper, data: Image.Image) -> yaml.Node:
    return dumper.represent_str("[image]")

yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_representer(Image.Image, image_representer)
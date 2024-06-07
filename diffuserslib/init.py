from diffuserslib.inference import DiffusersPipelines
from diffuserslib.functional import WorkflowRunner
from diffuserslib.GlobalConfig import GlobalConfig
from diffuserslib.functional.nodes.image.diffusers.RandomPromptProcessorNode import RandomPromptProcessorNode
from typing import List
import yaml
import os
import numpy as np
from diffuserslib.functional.types import Video, Audio, Vector
from PIL import Image, PngImagePlugin
import huggingface_hub


def initializeDiffusers(configs:List[str]=["config.yml"], modelconfigs:List[str]=["modelconfig.yml"], promptmods:List[str]=[]):
    safety_checker = True
    device = "cuda"
    models = "./models"
    for config in configs:
        if os.path.exists(config):
            configdata = yaml.safe_load(open(config, "r"))
            device = configdata["device"]
            safety_checker = configdata["safety"]
            if("hf_token" in configdata):
                huggingface_hub.login(configdata["hf_token"])
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
            if ("folders" not in configdata):
                continue
            if ("embeddings" in configdata["folders"]):
                embeddings = configdata["folders"]["embeddings"]
                GlobalConfig.embeddings_dirs.extend(embeddings)
                for embeddingdir in embeddings:
                    print("load embeddings from: ", embeddingdir)
                    DiffusersPipelines.pipelines.loadTextEmbeddings(embeddingdir)
            if ("loras" in configdata["folders"]):
                loras = configdata["folders"]["loras"]
                GlobalConfig.loras_dirs.extend(loras)
                for loradir in loras:
                    DiffusersPipelines.pipelines.loadLORAs(loradir)

    for promptmod in promptmods:
        RandomPromptProcessorNode.loadModifierDictFile(promptmod)



# Initialise yaml representers for yml dumps
def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())
def image_representer(dumper: yaml.Dumper, data: Image.Image) -> yaml.Node:
    return dumper.represent_str("[image]")
def video_representer(dumper: yaml.Dumper, data: Video) -> yaml.Node:
    return dumper.represent_str("[video]")
def audio_representer(dumper: yaml.Dumper, data: Audio) -> yaml.Node:
    return dumper.represent_str("[audio]")
def vector_representer(dumper: yaml.Dumper, data: Vector) -> yaml.Node:
    return dumper.represent_list(data.coordinates)

yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_representer(Image.Image, image_representer)
yaml.add_representer(PngImagePlugin.PngImageFile, image_representer)
yaml.add_representer(Video, video_representer)
yaml.add_representer(Audio, audio_representer)
yaml.add_representer(Vector, vector_representer)

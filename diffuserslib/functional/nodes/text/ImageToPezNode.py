from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.util.ModuleLoader import ModuleLoader
import sys
import argparse


class ImageToPezNode(FunctionalNode):

    CLIP_MODELS = {
        "sd_1_5": {
            "model": "ViT-L-14",
            "pretrain": "openai"
        },
        "sd_2_0": {
            "model": "ViT-H-14",
            "pretrain": "laion2b_s32b_b79k"
        },
        "sdxl": {
            "model": "ViT-bigG-14",
            "pretrain": "laion2b_s39b_b160k"
        }
    }

    open_clip = None
    optim_utils = None


    def __init__(self, 
                 image:ImageFuncType,
                 clip_model:StringFuncType = "sdxl",
                 iterations:IntFuncType = 100,
                 name:str = "image_to_pez",):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("clip_model", clip_model, str)
        self.addParam("iterations", iterations, int)
        if not ImageToPezNode.open_clip and not ImageToPezNode.optim_utils:
            ModuleLoader.load_from_directory("workspace/hard-prompts-made-easy/open_clip", recursive=True)
            ModuleLoader.load_from_file("workspace/hard-prompts-made-easy/optim_utils.py", module_name="optim_utils")
            ImageToPezNode.open_clip = sys.modules["open_clip"]
            ImageToPezNode.optim_utils = sys.modules["optim_utils"]
        

    def process(self, image:Image.Image, clip_model:str, iterations:int) -> str:
        assert ImageToPezNode.open_clip and ImageToPezNode.optim_utils, "open_clip and optim_utils modules not loaded"

        args = argparse.Namespace()
        args.__dict__["iter"] = iterations
        args.__dict__["prompt_len"] = 16
        args.__dict__["lr"] = 0.1
        args.__dict__["weight_decay"] = 0.1
        args.__dict__["prompt_bs"] = 1
        args.__dict__["loss_weight"] = 1.0
        args.__dict__["print_step"] = 100
        args.__dict__["batch_size"] = 1
        args.__dict__["clip_model"] = self.CLIP_MODELS[clip_model]["model"]
        args.__dict__["clip_pretrain"] = self.CLIP_MODELS[clip_model]["pretrain"]

        model, _, preprocess = ImageToPezNode.open_clip.create_model_and_transforms(args.__dict__["clip_model"], args.__dict__["clip_pretrain"], device="mps")
        learned_prompt = ImageToPezNode.optim_utils.optimize_prompt(model, preprocess, args, "mps", target_images=[image])

        return learned_prompt

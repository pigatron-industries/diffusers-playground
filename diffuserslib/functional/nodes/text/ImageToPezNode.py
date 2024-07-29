from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.util.ModuleLoader import ModuleLoader
import sys
import argparse


class ImageToPezNode(FunctionalNode):
    def __init__(self, 
                 image:ImageFuncType,
                 name:str = "image_to_pez"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        

    def process(self, image:Image.Image) -> str:
        ModuleLoader.load_from_directory("workspace/hard-prompts-made-easy/open_clip", recursive=True)
        ModuleLoader.load_from_file("workspace/hard-prompts-made-easy/optim_utils.py", module_name="optim_utils")
        open_clip = sys.modules["open_clip"]
        optim_utils = sys.modules["optim_utils"]

        clip_model = 'ViT-H-14'
        clip_pretrain = 'laion2b_s32b_b79k'

        args = argparse.Namespace()
        args.__dict__["iter"] = 100
        args.__dict__["prompt_len"] = 16
        args.__dict__["lr"] = 0.1
        args.__dict__["weight_decay"] = 0.1
        args.__dict__["prompt_bs"] = 1
        args.__dict__["loss_weight"] = 1.0
        args.__dict__["print_step"] = 100
        args.__dict__["batch_size"] = 1
        args.__dict__["clip_model"] = clip_model
        args.__dict__["clip_pretrain"] = clip_pretrain

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrain, device="mps")
        learned_prompt = optim_utils.optimize_prompt(model, preprocess, args, "mps", target_images=[image])

        return learned_prompt

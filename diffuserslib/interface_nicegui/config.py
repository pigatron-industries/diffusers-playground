from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional import *
from typing import List

selectable_nodes_config:List[FunctionalNode] = [
    RandomIntNode(min_max = MinMaxIntInputNode(), name = "Random Integer"), 
    RandomFloatNode(min_max = MinMaxFloatInputNode(), name = "Random Float"),
    RandomImageNode(paths = FileSelectInputNode(), name = "Random image"),
    ImageDiffusionNode(models = DiffusionModelUserInputNode(), 
                       size = SizeUserInputNode(value = (512, 512)), 
                       prompt = TextAreaInputNode(value = "", name="prompt"),
                       negprompt = StringUserInputNode(value = "", name="negprompt"),
                       steps = IntUserInputNode(value = 20, name = "steps"),
                       cfgscale = FloatUserInputNode(value = 7.0, name = "cfgscale"),
                       scheduler = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler"),
                       name = "Image Diffusion"),
]

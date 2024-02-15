from diffuserslib.functional import *

def name():
    return "Image Diffusion"

def build():
    model_input = DiffusionModelUserInputNode(name = "model")
    size_input = SizeUserInputNode(value = (512, 512), name = "size")
    prompt_input = TextAreaInputNode(value = "", name = "prompt")
    negprompt_input = TextAreaInputNode(value = "", name = "negprompt")
    steps_input = IntUserInputNode(value = 40, name = "steps")
    cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
    seed_input = IntUserInputNode(value = None, name = "seed")
    scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", 
                                              name = "scheduler",
                                              options = ["DPMSolverMultistepScheduler",
                                                         "EulerDiscreteScheduler", 
                                                         "EulerAncestralDiscreteScheduler"])
    conditioning_inputs = ListUserInputNode(type = ConditioningInputType, name = "conditioning_inputs")

    diffusion = ImageDiffusionNode(models = model_input,
                                   size = size_input,
                                   prompt = prompt_input,
                                   negprompt = negprompt_input,
                                   steps = steps_input,
                                   cfgscale = cfgscale_input,
                                   seed = seed_input,
                                   scheduler = scheduler_input,
                                   conditioning_inputs = conditioning_inputs)
    return diffusion

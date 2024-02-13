from diffuserslib.functional import *

def name():
    return "Image Diffusion"

def build():
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

    # TODO - Add support for conditioning inputs

    diffusion = ImageDiffusionNode(models = [ModelParameters("digiplay/Juggernaut_final")],
                                   size = size_input,
                                   prompt = prompt_input,
                                   negprompt = negprompt_input,
                                   steps = steps_input,
                                   cfgscale = cfgscale_input,
                                   seed = seed_input,
                                   scheduler = scheduler_input,
                                   conditioning_inputs = [])
    return diffusion

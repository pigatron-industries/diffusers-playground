from diffuserslib.functional import *


class RandomIntWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random Integer", int, workflow=False, subworkflow=True)

    def build(self):
        min_max = MinMaxIntInputNode()
        random_int = RandomIntNode(min_max = min_max)
        return random_int
    

class RandomFloatWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random Float", float, workflow=False, subworkflow=True)

    def build(self):
        min_max = MinMaxFloatInputNode()
        random_float = RandomFloatNode(min_max = min_max)
        return random_float


class RandomImageWorkflow(WorkflowBuilder):
    
    def __init__(self):
        super().__init__("Random Image", Image.Image, workflow=False, subworkflow=True)

    def build(self):
        paths = FileSelectInputNode()
        random_image = RandomImageNode(paths = paths)
        return random_image


class ImageDiffusionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion", Image.Image, workflow=True, subworkflow=True)

    def build(self):
        models = DiffusionModelUserInputNode()
        size = SizeUserInputNode(value = (512, 512))
        prompt = TextAreaInputNode(value = "", name="prompt")
        negprompt = StringUserInputNode(value = "", name="negprompt")
        steps = IntUserInputNode(value = 20, name = "steps")
        cfgscale = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")
        image_diffusion = ImageDiffusionNode(models = models, 
                                            size = size, 
                                            prompt = prompt,
                                            negprompt = negprompt,
                                            steps = steps,
                                            cfgscale = cfgscale,
                                            scheduler = scheduler)
        return image_diffusion

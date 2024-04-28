from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import *


class ImageDiffusionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion", Image.Image, workflow=True, subworkflow=True)

    def build(self):
        models_input = DiffusionModelUserInputNode()
        size_input = SizeUserInputNode(value = (512, 512))
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        image_diffusion = ImageDiffusionNode(models = models_input, 
                                            size = size_input, 
                                            prompt = prompt_processor,
                                            negprompt = negprompt_input,
                                            seed = seed_input,
                                            steps = steps_input,
                                            cfgscale = cfgscale_input,
                                            scheduler = scheduler_input)
        
        models_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))
        return image_diffusion
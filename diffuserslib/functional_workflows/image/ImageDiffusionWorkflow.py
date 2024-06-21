from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import *
from diffusers.schedulers import AysSchedules


class ImageDiffusionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion", Image.Image, workflow=True, subworkflow=True)

    def build(self):
        models_input = DiffusionModelUserInputNode()
        lora_input = LORAModelUserInputNode(diffusion_model_input = models_input, name = "lora")
        size_input = SizeUserInputNode(value = (512, 512))
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")
        clipskip_input = IntUserInputNode(value = None, name = "clipskip")

        sigmas_options = {
            "None": None,
            "StableDiffusion-AYS-10Step-Sigmas": AysSchedules["StableDiffusionSigmas"],
            "StableDiffusionXL-AYS-10Step-Sigmas": AysSchedules["StableDiffusionXLSigmas"],
        }
        sigmas_input = DictSelectUserInputNode(options = sigmas_options, value = None, name = "sigmas")
        sigmas_input.addUpdateListener(lambda: steps_input.setValue(len(sigmas_input.getSelectedOption())-1) if sigmas_input.getSelectedOption() is not None else None)

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        models_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))

        image_diffusion = ImageDiffusionNode(models = models_input, 
                                            loras = lora_input,
                                            size = size_input, 
                                            prompt = prompt_processor,
                                            negprompt = negprompt_input,
                                            seed = seed_input,
                                            steps = steps_input,
                                            cfgscale = cfgscale_input,
                                            scheduler = scheduler_input,
                                            clipskip = clipskip_input,
                                            sigmas = sigmas_input)
        return image_diffusion
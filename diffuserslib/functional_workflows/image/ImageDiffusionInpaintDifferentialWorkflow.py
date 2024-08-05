from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import *


class ImageDiffusionInpaintWorkflow(WorkflowBuilder):
    """
    Inpaint masked areas of an image using an inpaint model, then a differential model to blend better with the original.
    This is done because the SDXL inpaint models don't match colours well with the original image.
    """

    def __init__(self):
        super().__init__("Image Diffusion - Inpaint - Differential", Image.Image, workflow=True, subworkflow=False)

    def build(self):
        # Inpaint inputs
        image_input = ImageUploadInputNode()
        inpaint_scale_input = FloatUserInputNode(value = 1.0, name = "inpaint_scale")
        inpaint_model_input = DiffusionModelUserInputNode(modeltype = "inpaint", name="inpaint_model")
        loras_input = LORAModelUserInputNode(diffusion_model_input = inpaint_model_input, name = "lora")
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")

        # Inpaint conditioning images
        init_condition = ConditioningInputNode(image = image_input, model = "initimage", scale = inpaint_scale_input, name = "init_condition")
        maskimage = ImageAlphaToMaskNode(image = image_input, smooth = False, name = "maskimage")
        mask_condition = ConditioningInputNode(image = maskimage, model = "maskimage", name = "inpaint_mask_condition")

        # Inpaint diffusion
        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        inpaint_model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(inpaint_model_input.basemodel)))
        image_diffusion = ImageDiffusionNode(models = inpaint_model_input,
                                    loras = loras_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = [ init_condition, mask_condition ], name="inpaint_diffusion")
        
        mask_dilate = MaskDilationNode(mask = maskimage, dilation = 10, feather = 10, name = "inpaint_mask_dilate")
        image_composite = ImageCompositeNode(foreground = image_diffusion, background = image_input, mask = mask_dilate, name = "composite")

        # ======= DIFFERENTIAL PASS =======

        # Differential pass inputs
        generate_model_input = DiffusionModelUserInputNode(modeltype = "generate", name="differential_model")
        diff_scale_input = FloatUserInputNode(value = 0.2, name = "differential_scale")
        dilation_input = IntUserInputNode(value = 20, name = "differential_dilation")

        # Differential pass conditioning images
        diffinit_condition = ConditioningInputNode(image = image_composite, model = "initimage", scale = diff_scale_input, name = "differential_init_condition")
        diffmask_dilate = MaskDilationNode(mask = maskimage, dilation = dilation_input, feather = dilation_input, name = "differential_mask_dilate")
        diffmask_condition = ConditioningInputNode(image = diffmask_dilate, model = "diffmaskimage", name = "differential_mask_condition")

        # Differential pass diffusion
        differential_diffusion = ImageDiffusionNode(models = generate_model_input,
                            loras = loras_input,
                            prompt = prompt_processor,
                            negprompt = negprompt_input,
                            steps = steps_input,
                            cfgscale = cfgscale_input,
                            seed = seed_input,
                            scheduler = scheduler_input,
                            conditioning_inputs = [ diffinit_condition, diffmask_condition ], name="differential_diffusion")
        
        return differential_diffusion
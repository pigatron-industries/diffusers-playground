from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.ImageLanguageModelCompletionNode import ImageLanguageModelCompletionNode


class ImageLanguageModelCompletionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Image Language Model Completion", str, workflow=True, subworkflow=True)


    def build(self):
        models = [""]
        try:
            models = list(OllamaModels.loadLocalModels().keys())
        except:
            print("Error loading Ollama models. Is Ollama running?")
            pass
        model_input = ListSelectUserInputNode(value = "llava:13b", options = models, name = "model_input")
        prompt_input = TextAreaInputNode(value = "", name = "prompt_input")
        images_input = ImageUploadInputNode(name = "images_input")

        llm = ImageLanguageModelCompletionNode(model=model_input, prompt=prompt_input, images=images_input, name="llm")
        return llm

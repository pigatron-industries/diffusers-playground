from diffuserslib.functional.nodes.text.llm import LanguageModelCompletionNode
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelCompletionNode import LanguageModelCompletionNode


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
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model_input")
        prompt_input = TextAreaInputNode(value = "", name = "prompt_input")

        llm = LanguageModelCompletionNode(model=model_input, prompt=prompt_input, name="llm")
        return llm

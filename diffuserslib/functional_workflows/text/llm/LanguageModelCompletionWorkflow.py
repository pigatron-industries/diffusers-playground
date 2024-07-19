from diffuserslib.functional.nodes.text.llm import LanguageModelCompletionNode
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelCompletionNode import LanguageModelCompletionNode


class LanguageModelCompletionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Language Model Completion", str, workflow=True, subworkflow=True)


    def build(self):
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = list(OllamaModels.modelList.keys()), name = "model_input")
        prompt_input = TextAreaInputNode(value = "", name = "prompt_input")

        llm = LanguageModelCompletionNode(model=model_input, prompt=prompt_input, name="llm")
        return llm

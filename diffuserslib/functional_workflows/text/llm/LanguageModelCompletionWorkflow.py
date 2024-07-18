from diffuserslib.functional.nodes.text.llm import LanguageModelCompletionNode
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.LanguageModelCompletionNode import LanguageModelCompletionNode


class LanguageModelCompletionWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Language Model Completion", str, workflow=True, subworkflow=True)


    def build(self):
        prompt_input = TextAreaLinesInputNode(value = "", name = "prompt_input")

        llm = LanguageModelCompletionNode(prompt=prompt_input, name="llm")
        return llm

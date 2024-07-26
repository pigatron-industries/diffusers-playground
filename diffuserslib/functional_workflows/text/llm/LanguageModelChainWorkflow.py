from diffuserslib.functional.nodes.text import TemplateNode
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.TemplateNode import TemplateNode
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelChatNode import LanguageModelChatNode
from diffuserslib.functional.nodes.text.llm.LanguageModelCompletionNode import LanguageModelCompletionNode
from diffuserslib.functional.nodes.text.llm.ChatMessageInputNode import ChatMessageInputNode, ChatHistoryInputNode
from llama_index.core.llms import ChatMessage


class LanguageModelChainWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Language Model Chain", str, workflow=True, converse=True)


    def build(self):
        models = [""]
        try:
            models = list(OllamaModels.loadLocalModels().keys())
        except:
            print("Error loading Ollama models. Is Ollama running?")
            pass
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model_input")
        message_input = TextAreaInputNode(value = "", name = "message_input")

        llm_out1 = LanguageModelCompletionNode(model=model_input, prompt=message_input, name="llm")

        llm_out2 = LanguageModelCompletionNode(model=model_input, prompt=llm_out1, name="llm")

        return llm_out2
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelChatNode import LanguageModelChatNode
from diffuserslib.functional.nodes.text.llm.ChatMessageInputNode import ChatMessageInputNode, ChatHistoryInputNode


class LanguageModelChatWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Language Model Chat", str, workflow=True, converse=True)


    def build(self):
        models = list(OllamaModels.loadLocalModels().keys())
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model")
        history_input = ChatHistoryInputNode(value = [], name = "history")
        message_input = ChatMessageInputNode(value = None, name = "message")

        llm = LanguageModelChatNode(model=model_input, message=message_input, history=history_input, name="chat")
        return llm

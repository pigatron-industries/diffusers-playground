from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelChatNode import LanguageModelChatNode
from diffuserslib.functional.nodes.text.llm.ChatMessageInputNode import ChatMessageInputNode, ChatHistoryInputNode
from llama_index.core.llms import ChatMessage


class LanguageModelChatWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Language Model Chat", ChatMessage, workflow=True, converse=True)


    def build(self):
        models = [""]
        try:
            models = list(OllamaModels.loadLocalModels().keys())
        except:
            print("Error loading Ollama models. Is Ollama running?")
            pass
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model")
        system_input = TextAreaInputNode(value = "", name = "system")
        history_input = ChatHistoryInputNode(value = [], name = "history")
        message_input = ChatMessageInputNode(value = None, name = "message")
        temperature_input = FloatUserInputNode(value = 0.75, name = "temperature")

        llm = LanguageModelChatNode(model=model_input, message=message_input, history=history_input, system_prompt=system_input, 
                                    temperature=temperature_input, name="chat")
        return llm

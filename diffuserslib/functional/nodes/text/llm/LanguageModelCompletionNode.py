from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from llama_index.llms.ollama import Ollama


class LanguageModelCompletionNode(FunctionalNode):
    def __init__(self, 
                 prompt:StringFuncType,
                 model:StringFuncType = "llama3:8b",
                 name:str = "chat_completion"):
        super().__init__(name)
        self.addParam("prompt", prompt, str)
        self.addParam("model", model, str)
        self.model = None


    def process(self, prompt:str, model:str) -> str:
        if(model != self.model):
            self.model = model
            self.llm = Ollama(model = model, request_timeout = 120)
        response = self.llm.complete(prompt)
        return response.text

        




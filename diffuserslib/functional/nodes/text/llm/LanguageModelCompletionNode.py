from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from llama_index.llms.ollama import Ollama


class LanguageModelCompletionNode(FunctionalNode):
    def __init__(self, 
                 prompt:StringFuncType,
                 model:StringFuncType = "llama3:8b",
                 name:str = "llm_completion"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("prompt", prompt, str)
        self.model = None
        self.text = ""


    def process(self, model:str, prompt:str) -> str:
        self.stop_flag = False
        if(model != self.model):
            self.model = model
            self.llm = Ollama(model = model, request_timeout = 120)
        response = self.llm.stream_complete(prompt)
        self.text = ""
        for r in response:
            self.text = r.text
            if(self.stop_flag):
                print("LanguageModelChatNode: interrupted")
                break

        # print("LanguageModelCompletionNode: ", self.text)
        return self.text
    
    
    def progress(self) -> WorkflowProgress|None:
        return WorkflowProgress(0, self.text)
        

    def stop(self):
        self.stop_flag = True
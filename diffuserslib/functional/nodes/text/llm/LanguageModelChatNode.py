from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from .ChatMessageInputNode import *


class LanguageModelChatNode(FunctionalNode):
    def __init__(self, 
                 message:ChatMessageFuncType,
                 history:ChatHistoryFuncType,
                 model:StringFuncType = "llama3:8b",
                 system_prompt:StringFuncType = "",
                 temperature:FloatFuncType = 1.0,
                 name:str = "llm_chat"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("message", message, ChatMessage)
        self.addParam("history", history, List[ChatMessage])
        self.addParam("system_prompt", system_prompt, str)
        self.addParam("temperature", temperature, float)
        self.model = None
        self.response_message = None
        self.stop_flag = False


    def process(self, model:str, message:ChatMessage, history:List[ChatMessage], system_prompt:str, temperature:float) -> ChatMessage|None:
        self.stop_flag = False
        if(model != self.model or temperature != self.llm.temperature):
            self.model = model
            self.llm = Ollama(model = model, request_timeout = 120, temperature = temperature)

        messages = []
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        for histmessage in history:
            if(histmessage is not None and histmessage.content != ""):
                messages.append(histmessage)
        if(message is not None and message.content != ""):
            messages.append(message)

        self.response_message = None
        response = self.llm.stream_chat(messages)
        for r in response:
            self.response_message = r.message
            if(self.callback_progress is not None):
                self.callback_progress(WorkflowProgress(0, self.response_message))
            if(self.stop_flag):
                print("LanguageModelChatNode: interrupted")
                break

        print("LanguageModelChatNode:process end")
        print(self.response_message)
        return self.response_message
    
    
    def getProgress(self) -> WorkflowProgress|None:
        return WorkflowProgress(0, self.response_message)
        

    def stop(self):
        self.stop_flag = True
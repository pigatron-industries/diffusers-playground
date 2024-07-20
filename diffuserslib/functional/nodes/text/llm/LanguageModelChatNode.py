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
                 name:str = "llm_chat"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("message", message, ChatMessage)
        self.addParam("history", history, List[ChatMessage])
        self.addParam("system_prompt", system_prompt, str)
        self.model = None
        self.response_message = None
        self.stop_flag = False


    def process(self, model:str, message:ChatMessage, history:List[ChatMessage], system_prompt:str) -> ChatMessage|None:
        self.stop_flag = False
        if(model != self.model):
            self.model = model
            self.llm = Ollama(model = model, request_timeout = 120)

        messages = []
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        for message in history:
            if(message is not None and message.content != ""):
                messages.append(message)
        if(message is not None and message.content != ""):
            messages.append(message)

        print("LanguageModelChatNode:process start")
        print(messages)

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
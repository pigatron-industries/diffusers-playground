from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from llama_index.core.llms import ChatMessage
from typing import Callable, List
import yaml


ChatMessageFuncType = ChatMessage | Callable[[], ChatMessage]
ChatHistoryFuncType = List[ChatMessage] | Callable[[], List[ChatMessage]]


def chat_message_representer(dumper, data:ChatMessage):
    return dumper.represent_scalar(u'!ChatMessage', str(data.role.value) + "|" + str(data.content))
def chat_message_constructor(loader, node):
    value = loader.construct_scalar(node)
    role,sep,message = value.partition("|")
    return ChatMessage(role=role, content=message)
yaml.add_representer(ChatMessage, chat_message_representer)
yaml.add_constructor(u'!ChatMessage', chat_message_constructor)


class ChatMessageInputNode(UserInputNode):
    def __init__(self, value:ChatMessage|None=None, name:str="chat_message_input"):
        self.value = value
        super().__init__(name)

    def getValue(self) -> ChatMessage|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        pass

    def process(self) -> ChatMessage|None:
        return self.value
    


class ChatHistoryInputNode(UserInputNode):
    def __init__(self, value:List[ChatMessage], name:str="chat_history_input"):
        self.value = value
        super().__init__(name)

    def getValue(self) -> List[ChatMessage]:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        pass

    def process(self) -> List[ChatMessage]:
        return self.value
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class FeedbackNode(FunctionalNode):
    def __init__(self, 
                 init_value:Any,
                 input:FunctionalNode|None = None,
                 name:str = "feedback"):
        super().__init__(name)
        self.input = input
        self.init_value = None
        self.addInitParam("init_value", init_value, Any)


    def init(self, init_value:Any):
        print("FeedbackNode init")
        self.init_value = init_value


    def setInput(self, input:FunctionalNode):
        self.input = input


    def process(self) -> Any:
        if self.input is None:
            raise Exception("Feedback input not set")
        value = self.input.getPreviousOutput()
        if value is None:
            return self.init_value
        else:
            return value
    
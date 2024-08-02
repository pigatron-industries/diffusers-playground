from typing import Any
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import StringFuncType


class ConditionalNode(FunctionalNode):

    def __init__(self, condition:StringFuncType, name: str = "condition", **kwargs):
        super().__init__(name)
        self.addParam("condition", condition, str)
        for key, value in kwargs.items():
            self.addParam(key, value, Any)


    def evaluateParams(self):
        paramvalues = {}
        condition = self.params["condition"]

        if(callable(condition.value)):
            conditionvalue = condition.value()
        else:
            conditionvalue = condition.value
        selectedparam = self.params[str(conditionvalue).lower()]

        if(callable(selectedparam.value)):
            paramvalues["returnvalue"] = selectedparam.value()
        else:
            paramvalues["returnvalue"] = selectedparam.value
        return paramvalues
        

    def process(self, returnvalue:Any) -> Any:
        return returnvalue


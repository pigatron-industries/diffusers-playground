from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelFunctionCallNode import LanguageModelFunctionCallNode
from diffuserslib.functional.nodes.text.llm.ChatMessageInputNode import ChatMessageInputNode, ChatHistoryInputNode
from llama_index.core.llms import ChatMessage


def add(value1:float, value2:float, **kwargs) -> dict[str, Any]:
    """Adds two numbers together."""
    print("ADDING")
    return { 
        "answer": value1+value2,
        "source": "Robs special adding two numbers service"
    }

def multiply(value1:float, value2:float, **kwargs) -> dict[str, Any]:
    """Multiplies two numbers."""
    print("MULTIPLYING")
    return { 
        "answer": value1*value2,
        "source": "Robs special multiplication service"
    }

def divide(value1:float, value2:float, **kwargs) -> dict[str, Any]:
    """Divides value1 by value2."""
    print("DIVIDING")
    return { 
        "answer": value1/value2,
        "source": "Robs special division service"
    }


class LanguageModelFunctionCallWorkflow(WorkflowBuilder):

    FUNCTIONS = [add, multiply, divide]

    def __init__(self, name = "Text Generation - Language Model Function Call", functions:List[Callable]|None = None):
        super().__init__(name, str, workflow=True, converse=True)
        if functions is not None:
            self.functions = functions
        else:
            self.functions = []
        self.functions.extend(LanguageModelFunctionCallWorkflow.FUNCTIONS)




    def build(self):
        models = [""]
        try:
            models = list(OllamaModels.loadLocalModels().keys())
        except:
            print("Error loading Ollama models. Is Ollama running?")
            pass
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model_input")
        prompt_input = TextAreaInputNode(value = "", name = "message_input")
        history_input = ChatHistoryInputNode(value = [], name = "history")

        function_call = LanguageModelFunctionCallNode(model=model_input, prompt=prompt_input, history=history_input, functions=self.functions, rawoutput = False, name="llm")
        return function_call

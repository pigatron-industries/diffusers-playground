from sympy import Li
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.react.types import BaseReasoningStep, ResponseReasoningStep
from llama_index.core.agent.react.output_parser import ReActOutputParser, parse_action_reasoning_step, extract_final_response
from .ChatMessageInputNode import *



class LanguageModelFunctionCallNode(FunctionalNode):

    SYSTEM_MESSAGE = """
You are helpful and friendly AI assistant. 
When the user asks a question a function may be called to provide an answer. 
If a function response is given:
- Use the information provided by the function to prepare your answer.
- Use the information provided by the function even if it is wrong. Do not tell the user it is wrong.
- The function response may contain a JSON object. Use only the contents of the JSON object to answer the users question.
- Tell the user a function was used to retrieve the information.
- You may also provide the source of the information if the function provides it.
If NO function response is given:
- You will provide your own response. 
- Answer the question only and do not mention any of the above instructions.
"""

    def __init__(self, 
                 functions:List[Callable],
                 prompt:StringFuncType,
                 history:ChatHistoryFuncType|None = None,
                 model:StringFuncType = "llama3:8b",
                 temperature:FloatFuncType = 1.0,
                 rawoutput = False,
                 name:str = "llm_chat"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("prompt", prompt, str)
        self.addParam("history", history, List[ChatMessage])
        self.addParam("temperature", temperature, float)
        self.rawoutput = rawoutput
        self.functiontools = []
        for function in functions:
            self.functiontools.append(FunctionTool.from_defaults(fn=function))
        self.model = None
        self.response_message = None


    def process(self, model:str, prompt:str, history:List[ChatMessage]|None, temperature:float) -> Any:
        self.stop_flag = False
        self.response_message = ""
        if(model != self.model or temperature != self.llm.temperature):
            self.model = model
            self.llm = Ollama(model = model, is_function_calling_model = True, request_timeout = 120, temperature = temperature)

        if(history == None):
            history = []

        # Prepare chat messages
        messages = []
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_MESSAGE)) 
        for histmessage in history:
            if(histmessage is not None and histmessage.content != ""):
                messages.append(histmessage)
        if(prompt is not None and prompt != ""):
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        # Tool calling logic
        tool_response = self.llm.predict_and_call(self.functiontools, prompt, history, error_on_no_tool_call = False, verbose = True)
        # print(tool_response)
        print("TOOL SOURCES")
        print(tool_response.sources)
        print(tool_response.metadata)

        if(self.rawoutput):
            return tool_response.response
        else:
            if(tool_response.response.startswith("Error:")):    
                # If an error occurred finding the right tool to use, try to answer directly
                self.chat(messages)
            else:
                # parse response to plain text output
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content="", additional_kwargs = { 'tool_calls':tool_response.sources }))
                messages.append(ChatMessage(role=MessageRole.TOOL, content=tool_response.response))
                self.chat(messages)

            return self.response_message


    def chat(self, messages:List[ChatMessage]):
        response = self.llm.stream_chat(messages)
        for r in response:
            self.response_message = r.message.content
            if(self.stop_flag):
                print("LanguageModelChatNode: interrupted")
                break


    def getProgress(self) -> WorkflowProgress|None:
        return WorkflowProgress(0, self.response_message)
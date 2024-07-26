from diffuserslib.interface.WorkflowController import WorkflowController
from typing import Dict
from llama_index.core.llms import ChatMessage, MessageRole
from diffuserslib.functional.WorkflowRunner import WorkflowRunner



class ChatController(WorkflowController):

    instance = None
    messageid = 0

    @classmethod
    def get_instance(cls, history_filename: str):
        if cls.instance is None:
            cls.instance = ChatController(history_filename)
        return cls.instance

    def __init__(self, history_filename: str):
        super().__init__(history_filename)
        self.message_history:Dict[int, ChatMessage|None] = {}
        self.batchid = None
        self.lastmessageid = None


    def appendMessage(self, message):
        ChatController.messageid += 1
        self.message_history[ChatController.messageid] = message
        return ChatController.messageid
        

    def runWorkflow(self):
        self.batchid = super().runWorkflow()
        self.lastmessageid = self.appendMessage(ChatMessage(role=MessageRole.ASSISTANT, content="Thinking..."))
        

    def updateProgress(self):
        assert WorkflowRunner.workflowrunner is not None
        if(self.batchid is not None and self.lastmessageid is not None):
            WorkflowRunner.workflowrunner.getProgress()
            batch = WorkflowRunner.workflowrunner.getBatch(self.batchid)
            if(batch is not None):
                for runid, rundata in batch.rundata.items():
                    # assumes only a batch of 1
                    if(rundata.getStatus() == "Complete"):
                        self.message_history[self.lastmessageid] = ChatMessage(role=MessageRole.ASSISTANT, content=rundata.output)
                        WorkflowRunner.workflowrunner.removeBatch(batch.id)
                        self.batchid = None
                    elif(rundata.getStatus() == "Running"):
                        self.message_history[self.lastmessageid] = ChatMessage(role=MessageRole.ASSISTANT, content=rundata.progress.output)
                    elif(rundata.getStatus() == "Error"):
                        self.message_history[self.lastmessageid] = ChatMessage(role=MessageRole.ASSISTANT, content="Error: " + str(rundata.error))
                    else:
                        self.message_history[self.lastmessageid] = ChatMessage(role=MessageRole.ASSISTANT, content="Thinking...")
        return self.lastmessageid
    

    def removeMessage(self, id):
        if(id in self.message_history):
            del self.message_history[id]


    def clearChatHistory(self):
        if(self.model.workflow is not None):
            self.model.workflow.reset()
        self.message_history = {}
        # ChatController.messageid = 0
        # self.batchid = None
        # self.lastmessageid = None

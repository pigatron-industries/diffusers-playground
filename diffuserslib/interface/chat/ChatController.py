from diffuserslib.interface.WorkflowController import WorkflowController
from typing import Dict
from llama_index.core.llms import ChatMessage, MessageRole
from dataclasses import dataclass
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
        self.message_history:Dict[int, ChatMessage] = {}
        self.batchid = None


    def appendMessage(self, message):
        ChatController.messageid += 1
        self.message_history[ChatController.messageid] = message
        return ChatController.messageid
        

    def runWorkflow(self):
        self.batchid = super().runWorkflow()
        self.lastmessageid = self.appendMessage(ChatMessage(role=MessageRole.ASSISTANT, content="Generating..."))
        

    def updateProgress(self):
        assert WorkflowRunner.workflowrunner is not None
        if(self.batchid is not None and self.batchid in WorkflowRunner.workflowrunner.batchrundata):
            WorkflowRunner.workflowrunner.getProgress()
            batch = WorkflowRunner.workflowrunner.batchrundata[self.batchid]
            for runid, rundata in batch.rundata.items():
                # assumes only a batch of 1
                if(rundata.progress is not None):
                    self.message_history[self.lastmessageid] = rundata.progress.output
        return self.lastmessageid



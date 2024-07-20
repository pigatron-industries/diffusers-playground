from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes.text.llm.ChatMessageInputNode import ChatMessageInputNode, ChatHistoryInputNode
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from diffuserslib.interface.WorkflowController import WorkflowController
from ..WorkflowComponents import WorkflowComponents
from .ChatHistoryMessageControls import ChatHistoryMessageControls
from .ChatController import ChatController
from nicegui import ui
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict
from llama_index.core.llms import ChatMessage, MessageRole


default_output_width = 256


@dataclass
class ChatInput:
    text:str = ""



class ConverseInterfaceComponents(WorkflowComponents):

    def __init__(self, controller:ChatController):
        super().__init__(controller)
        self.controller = controller
        self.chat_input = ChatInput()
        self.history_controls:Dict[int, ChatHistoryMessageControls] = {}
        self.history_scroll = None
        self.scroll_bottom = True
        self.output = None
        self.timer = ui.timer(0.1, lambda: self.updateWorkflowProgress(), active=False)
        self.builders = {}
        for name, builder in self.controller.builders.items():
            if(builder.converse):
                self.builders[name] = builder.name
        self.builders = dict(sorted(self.builders.items(), key=lambda item: item[1]))
        self.message_node = None
        self.history_node = None
        self.findInputNodes()


    def loadWorkflow(self, workflow_name):
        super().loadWorkflow(workflow_name)
        self.findInputNodes()


    def findInputNodes(self):
        if(self.controller.model.workflow is not None):
            self.message_node = self.controller.model.workflow.getNodeByType(ChatMessageInputNode)
            self.history_node = self.controller.model.workflow.getNodeByType(ChatHistoryInputNode)
            # self.controller.model.workflow.setProgressCallback(self.updateWorkflowProgress)
            # self.controller.model.workflow.setFinishedCallback(self.finishedWorkflow)


    def runWorkflow(self):
        assert self.message_node is not None and self.history_node is not None
        message = ChatMessage(role=MessageRole.USER, content=self.chat_input.text)
        self.message_node.setValue(message)
        self.history_node.setValue(list(self.controller.message_history.values()))
        if(message.content != ""):
            self.appendMessage(message)
        self.timer.activate()
        self.controller.runWorkflow()
        self.chat_input.text = ""
        self.status.refresh()


    def appendMessage(self, message:ChatMessage):
        messageid = self.controller.appendMessage(message)
        with self.history_container:
            self.history_controls[messageid] = ChatHistoryMessageControls(messageid, message, self.controller)
        if(self.history_scroll is not None and self.scroll_bottom):
            self.scrollToBottom()


    def stopWorkflow(self):
        self.timer.deactivate()
        self.status.refresh()


    def updateWorkflowProgress(self):
        if(not self.controller.workflowrunner.running):
            self.timer.deactivate()
        self.status.refresh()

        messageid = self.controller.updateProgress()
        if(messageid is not None):
            message = self.controller.message_history[messageid]
            if(messageid in self.history_controls):
                # update existing messsage
                self.history_controls[messageid].update(message)
            else:
                # add new message
                with self.history_container:
                    self.history_controls[messageid] = ChatHistoryMessageControls(messageid, message, self.controller)
        if(self.history_scroll is not None and self.scroll_bottom):
            self.scrollToBottom()


    def finishedWorkflow(self):
        pass


    def clearHistory(self):
        if(self.controller.model.workflow is not None):
            self.controller.model.workflow.reset()
        self.controller.clearChatHistory()
        self.history_controls = {}
        self.outputs.refresh()
        self.status.refresh()

    
    @ui.refreshable
    def status(self):
        with ui.column().classes("gap-0").style("align-items:center;"):  #margin-top:1.3em;
            if(self.controller.isStopping()):
                ui.label("Stopping...").style("color: #ff0000;")
            elif(self.controller.isRunning()):
                ui.label("Generating...").style("color: #ffcc00;")
            else:
                ui.label("Idle").style("color: #00ff00;")

    
    def buttons(self):
        ui.button('Stop', on_click=lambda e: self.controller.stopWorkflow())
        ui.button('Clear', on_click=lambda e: self.clearHistory())


    @ui.refreshable
    def controls(self):
        self.workflowSelect(self.builders, ["str"])
        self.workflow_parameters()


    @ui.refreshable
    def outputs(self):
        with ui.splitter(horizontal=True, value=70).classes("w-full h-full no-wrap overflow-auto") as splitter:
            with splitter.before:
                self.history()
            with splitter.after:
                self.input()


    def input(self):
        with ui.column().classes("w-full h-full gap-0"):
            with ui.row().classes("w-full").style("background-color:#2b323b; padding: 0.5rem;"):
                with ui.column().classes("grow"):
                    pass
                with ui.column():
                    ui.button(icon='send', color='dark', on_click=self.runWorkflow)
            ui.textarea().bind_value(self.chat_input, 'text').classes('h-full-input w-full h-full grow').style("width: 100%; height: 100%;") \
                .on('keydown.enter', js_handler='(e) => { if(!e.shiftKey) e.preventDefault(); }').on('keyup.enter', self.onEnter)
            

    def onEnter(self, event):
        if(event.args['shiftKey']):
            return
        else:
            self.runWorkflow()


    def history(self):
        with ui.scroll_area(on_scroll = self.onScroll).classes("history-scroll w-full h-full") as self.history_scroll:
            with ui.column().classes("p-2 w-full") as self.history_container:
                for id, message in self.controller.message_history.items():
                    self.history_controls[id] = ChatHistoryMessageControls(id, message, self.controller)
                
    def onScroll(self, event):
        self.scroll_bottom = (event.vertical_size - event.vertical_container_size - event.vertical_position) < 50

    def scrollToBottom(self):
        assert self.history_scroll is not None
        self.history_scroll.scroll_to(percent=1.0)


    def settings(self):
        print(self.controller.message_history)
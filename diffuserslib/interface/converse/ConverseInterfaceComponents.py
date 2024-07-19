from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes.text.llm.ChatMessageInputNode import ChatMessageInputNode, ChatHistoryInputNode
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from diffuserslib.interface.WorkflowController import WorkflowController
from ..WorkflowComponents import WorkflowComponents
from nicegui import ui, run
from PIL import Image
from dataclasses import dataclass
from typing import List
from llama_index.core.llms import ChatMessage, MessageRole


default_output_width = 256


@dataclass
class ConverseInput:
    text:str = ""


@dataclass
class ConverseModel:
    input:ConverseInput
    history:List[ChatMessage]


class ConverseInterfaceComponents(WorkflowComponents):

    def __init__(self, controller:WorkflowController):
        super().__init__(controller)
        self.model = ConverseModel(input=ConverseInput(), history=[])
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


    async def runWorkflow(self):
        assert self.message_node is not None and self.history_node is not None
        message = ChatMessage(role=MessageRole.USER, content=self.model.input.text)
        self.message_node.setValue(message)
        self.model.history.append(message)
        self.history_node.setValue(self.model.history)
        self.timer.activate()
        self.running = True
        result = await run.io_bound(self.controller.runWorkflow)
        self.status.refresh()


    def stopWorkflow(self):
        self.timer.deactivate()
        self.status.refresh()


    def updateWorkflowProgress(self):
        if(not self.controller.workflowrunner.running):
            self.timer.deactivate()
        self.status.refresh()
        # for batchid, batchdata in self.controller.getBatchRunData().items():
        #     if(batchid in self.batchdata_controls):
        #         batchdata_container = self.batchdata_controls[batchid].batch_container
        #         for runid, rundata in batchdata.rundata.items():
        #             if(runid in self.rundata_controls):
        #                 # Update existing output controls
        #                 self.rundata_controls[runid].update()
        #             elif(batchdata_container is not None):
        #                 # Add new output controls
        #                 with batchdata_container:
        #                     self.workflow_output_rundata_container(batchid, runid)
        #     else:
        #          with self.outputs_container:
        #             self.workflow_output_batchdata(batchid) # type: ignore


    def finishedWorkflow(self):
        pass



    def clearHistory(self):
        if(self.controller.model.workflow is not None):
            self.controller.model.workflow.reset()
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
        ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size').bind_visibility_from(self.controller.model, 'run_type', value=1).style("width: 100px")
        ui.button('Clear', on_click=lambda e: self.clearHistory()).classes('align-middle')


    @ui.refreshable
    def controls(self):
        self.workflowSelect(self.builders, ["str"])
        self.workflow_parameters()


    @ui.refreshable
    def outputs(self):
        with ui.splitter(horizontal=True, value=70).classes("w-full h-full no-wrap overflow-auto") as splitter:
            with splitter.before:
                with ui.column().classes("p-2 w-full"):
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
            ui.textarea().bind_value(self.model.input, 'text').classes('h-full-input w-full h-full grow').style("width: 100%; height: 100%;").props('input-class=h-full')


    def history(self):
        pass
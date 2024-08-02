from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from diffuserslib.interface.WorkflowController import WorkflowController
from .ChatInterfaceComponents import ConverseInterfaceComponents
from .ChatController import ChatController
from dataclasses import dataclass



class ConverseView(AbstractView):

    splitter_position = 25

    controller = None

    @staticmethod
    def getControllerInstance():
        if ConverseView.controller is None:
            ConverseView.controller = ChatController("./.history_chat.yml")
        return ConverseView.controller


    @ui.page('/chat')
    def gui():
        app.on_exception(ui.notify)
        view = ConverseView()
        view.page()


    def __init__(self):
        self.interface_components = ConverseInterfaceComponents(self.getControllerInstance())


    def toggles(self):
        ui.label("Chat").style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')


    @ui.refreshable
    def status(self):
        self.interface_components.status()  # type: ignore


    @ui.refreshable
    def buttons(self):
        self.interface_components.buttons()  # type: ignore


    @ui.refreshable
    def controls(self):
        self.interface_components.controls()  # type: ignore
            

    @ui.refreshable
    def outputs(self):
        self.interface_components.outputs()  # type: ignore
   

    def settings(self):
        self.interface_components.settings()  # type: ignore
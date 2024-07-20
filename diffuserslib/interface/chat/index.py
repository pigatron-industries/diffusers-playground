from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from diffuserslib.interface.WorkflowController import WorkflowController
from .ChatInterfaceComponents import ConverseInterfaceComponents
from .ChatController import ChatController
from dataclasses import dataclass



class ConverseView(AbstractView):

    splitter_position = 25

    @ui.page('/chat')
    def gui():
        app.on_exception(ui.notify)
        view = ConverseView()
        view.page()


    def __init__(self):
        self.controller = ChatController.get_instance("./.history_chat.yml")
        self.interface_components = ConverseInterfaceComponents(self.controller)


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
        pass
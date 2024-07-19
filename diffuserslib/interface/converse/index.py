from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from diffuserslib.interface.WorkflowController import WorkflowController
from .ConverseInterfaceComponents import ConverseInterfaceComponents
from dataclasses import dataclass



class ConverseView(AbstractView):

    @ui.page('/chat')
    def gui():
        app.on_exception(ui.notify)
        view = ConverseView()
        view.page()


    def __init__(self):
        self.controller = WorkflowController.get_instance()
        self.interface_components = ConverseInterfaceComponents(self.controller)


    def toggles(self):
        ui.label("Chat").style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')


    @ui.refreshable
    def status(self):
        self.interface_components.status()  # type: ignore


    @ui.refreshable
    def buttons(self):
        pass


    @ui.refreshable
    def controls(self):
        self.interface_components.controls()  # type: ignore
            

    @ui.refreshable
    def outputs(self):
        self.interface_components.outputs()  # type: ignore
   

    def settings(self):
        pass
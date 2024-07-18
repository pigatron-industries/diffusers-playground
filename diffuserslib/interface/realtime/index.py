from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from diffuserslib.interface.WorkflowController import WorkflowController
from .RealtimeInterfaceComponents import RealtimeInterfaceComponents


class RealtimeView(AbstractView):

    @ui.page('/realtime')
    def realtime():
        app.on_exception(ui.notify)
        view = RealtimeView()
        view.page()


    def __init__(self):
        self.controller = WorkflowController.get_instance()
        self.interface_components = RealtimeInterfaceComponents(self.controller)


    def toggles(self):
        ui.label("Realtime").style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')


    @ui.refreshable
    def status(self):
        self.interface_components.status() # type: ignore


    @ui.refreshable
    def buttons(self):
        self.interface_components.buttons()


    def settings(self):
        if(self.controller.model.workflow is not None):
            self.controller.model.workflow.printDebug()
        with ui.dialog(value=True) as settings_dialog, ui.card():
            self.interface_components.settings()


    @ui.refreshable
    def controls(self):
        self.interface_components.controls() # type: ignore
            

    @ui.refreshable
    def outputs(self):
        self.interface_components.outputs() # type: ignore
   
            
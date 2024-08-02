from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from diffuserslib.interface.WorkflowController import WorkflowController
from ..WorkflowComponents import WorkflowComponents
from .BatchInterfaceComponents import BatchInterfaceComponents
from ..realtime.RealtimeInterfaceComponents import RealtimeInterfaceComponents


class BatchView(AbstractView):

    controller = None

    @staticmethod
    def getControllerInstance():
        if BatchView.controller is None:
            BatchView.controller = WorkflowController()
        return BatchView.controller
        

    @ui.page('/')
    def gui():
        app.on_exception(ui.notify)
        view = BatchView()
        view.page()

    @ui.page('/batch')
    def batch():
        app.on_exception(ui.notify)
        view = BatchView()
        view.page()


    def __init__(self):
        self.interface_components = BatchInterfaceComponents(BatchView.getControllerInstance())


    def toggles(self):
        ui.label("Batch").style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')


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
   
            
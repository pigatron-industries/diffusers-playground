from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from .BatchController import BatchController
from .InterfaceComponents import InterfaceComponents
from .BatchInterfaceComponents import BatchInterfaceComponents
from .RealtimeInterfaceComponents import RealtimeInterfaceComponents


class View(AbstractView):

    @ui.page('/')
    def gui():
        app.on_exception(ui.notify)
        view = View()
        view.page()

    @ui.page('/batch')
    def batch():
        app.on_exception(ui.notify)
        view = View()
        view.page()


    def __init__(self):
        self.controller = BatchController()
        self.setInterfaceComponents()


    def onUpdateRunType(self):
        self.setInterfaceComponents()
        self.status.refresh()
        self.buttons.refresh()
        self.controls.refresh()
        self.outputs.refresh()


    def setInterfaceComponents(self):
        if(self.controller.model.run_type == 1):
            self.interface_components = BatchInterfaceComponents(self.controller)
        elif(self.controller.model.run_type == 2):
            self.interface_components = RealtimeInterfaceComponents(self.controller)
        else:
            self.interface_components = InterfaceComponents(self.controller)


    def toggles(self):
        ui.toggle({1: 'Batch', 2: 'Realtime'}, on_change=self.onUpdateRunType).bind_value(self.controller.model, 'run_type').style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')


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
   
            
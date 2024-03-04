from nicegui import ui, context
from .api import *
from .Controller import Controller
from .InterfaceComponents import InterfaceComponents
from .BatchInterfaceComponents import BatchInterfaceComponents
from .RealtimeInterfaceComponents import RealtimeInterfaceComponents


class View:

    @ui.page('/')
    def gui():
        app.on_exception(ui.notify)
        view = View()
        view.page()


    def __init__(self):
        self.controller = Controller()
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
        elif(self.controller.model.run_type == 3):
            self.interface_components = RealtimeInterfaceComponents(self.controller)
        else:
            self.interface_components = InterfaceComponents(self.controller)


    def page(self):
        ui.page_title('Generative Toolkit')
        ui.label.default_classes('label')
        ui.select.default_classes('w-80')
        ui.query('body').style(f'background-color: rgb(24, 28, 33)')
        ui.add_head_html('''
            <style>
                .align-middle {
                    transform: translateY(50%);
                }
            </style>
        ''')
        context.get_client().content.classes('h-[100vh] p-0')

        with ui.column().classes("w-full h-full no-wrap gap-0"):
            with ui.row().classes("w-full p-2 place-content-between").style("background-color:#2b323b; border-bottom:1px solid #585b5f"):
                ui.toggle({1: 'Batch', 2: 'Animate', 3: 'Realtime'}, on_change=self.onUpdateRunType).bind_value(self.controller.model, 'run_type').style('margin-top:1.3em; margin-right:5em; margin-bottom:0.1em;')
                self.status() # type: ignore
                with ui.row():
                    self.buttons() # type: ignore
                    ui.button(icon='settings', color='dark', on_click=self.settings).classes('align-middle')
                    
            with ui.splitter(value=40).classes("w-full h-full no-wrap overflow-auto") as splitter:
                with splitter.before:
                    with ui.column().classes("p-2 w-full"):
                        self.controls() # type: ignore
                with splitter.after:
                    self.outputs() # type: ignore


    @ui.refreshable
    def status(self):
        self.interface_components.status() # type: ignore


    @ui.refreshable
    def buttons(self):
        self.interface_components.buttons()


    def settings(self):
        if(self.controller.workflow is not None):
            self.controller.workflow.printDebug()
        with ui.dialog(value=True) as settings_dialog, ui.card():
            self.interface_components.settings()


    @ui.refreshable
    def controls(self):
        self.interface_components.controls() # type: ignore
            

    @ui.refreshable
    def outputs(self):
        self.interface_components.outputs() # type: ignore
   
            
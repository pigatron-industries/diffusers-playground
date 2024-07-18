from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView
from diffuserslib.interface.WorkflowController import WorkflowController
from dataclasses import dataclass


@dataclass
class ConverseInput:
    text:str = ""


@dataclass
class ConverseModel:
    input:ConverseInput


class ConverseView(AbstractView):

    @ui.page('/converse')
    def gui():
        app.on_exception(ui.notify)
        view = ConverseView()
        view.page()


    def send(self):
        pass
    

    def __init__(self):
        self.model = ConverseModel(input=ConverseInput())
        self.controller = WorkflowController.get_instance()


    @ui.refreshable
    def status(self):
        pass


    @ui.refreshable
    def buttons(self):
        pass


    @ui.refreshable
    def controls(self):
        pass
            

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
                    ui.button(icon='send', color='dark', on_click=self.send)
            ui.textarea().bind_value(self.model.input, 'text').classes('h-full-input w-full h-full grow').style("width: 100%; height: 100%;").props('input-class=h-full')


    def history(self):
        pass
   

    def settings(self):
        pass
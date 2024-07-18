from nicegui import ui, app
from diffuserslib.interface.AbstractView import AbstractView


class ConverseView(AbstractView):

    @ui.page('/converse')
    def gui():
        app.on_exception(ui.notify)
        view = ConverseView()
        view.page()


    def __init__(self):
        pass
        # self.controller = Controller()


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
        pass
   

    def settings(self):
        pass
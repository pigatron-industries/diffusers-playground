from .Controller import Controller
from nicegui import ui


class InterfaceComponents:

    def __init__(self, controller:Controller):
        self.controller = controller


    def status(self):
        pass

    def buttons(self):
        pass

    def controls(self):
        pass

    def outputs(self):
        pass
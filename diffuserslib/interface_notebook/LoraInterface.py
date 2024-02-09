from IPython.display import display
from .WidgetHelpers import *


class LoraInterface:
    def __init__(self, interface):
        self.interface = interface
        self.lora_dropdown = dropdown(interface, label="LORA:", options=[""], value=None)
        self.loraweight_text = floatText(interface, label="LORA weight:", value=1)

    def display(self):
        display(self.lora_dropdown,
                self.loraweight_text)
        
    def hide(self):
        self.lora_dropdown.layout.display = 'none'
        self.loraweight_text.layout.display = 'none'

    def show(self):
        self.lora_dropdown.layout.display = 'flex'
        self.loraweight_text.layout.display = 'flex'
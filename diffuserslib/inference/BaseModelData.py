from .TextEmbedding import TextEmbeddings
from .LORA import LORA, LORAs
from typing import Dict


class BaseModelData:
    def __init__(self, base:str, textembeddings:TextEmbeddings, loras:LORAs|None=None, modifierdict:Dict[str, list[str]]|None = None):
        self.base:str = base
        self.textembeddings:TextEmbeddings = textembeddings
        if (loras is None):
            self.loras = LORAs()
        else:
            self.loras:LORAs = loras
        if (modifierdict is None):
            self.modifierdict = {}
        else:
            self.modifierdict = modifierdict

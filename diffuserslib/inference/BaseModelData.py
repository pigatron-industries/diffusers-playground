from .TextEmbedding import TextEmbeddings


class BaseModelData:
    def __init__(self, base : str, textembeddings : TextEmbeddings, modifierdict = None):  #: Dict[str, list[str]]
        self.base : str = base
        self.textembeddings : TextEmbeddings = textembeddings
        self.loras = {}
        if (modifierdict is None):
            self.modifierdict = {}
        else:
            self.modifierdict = modifierdict


class Argument:
        pass


class BatchArgument:
    pass


class PlaceholderArgument(Argument):
    def __init__(self, name):
        self.name = name
        self.value = None

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

    def __call__(self, **kwargs):
        if(self.value is None):
            raise Exception(f'Placeholder {self.name} not set')
        if(callable(self.value)):
            return self.value()
        else:
            return self.value
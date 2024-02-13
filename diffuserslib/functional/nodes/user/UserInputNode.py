from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from nicegui import ui


class UserInputNode(FunctionalNode):
    """ represents input from the user """
    def __init__(self, name:str="user_input"):
        super().__init__(name)
    def ui(self):
        raise Exception("Not implemented")


class IntUserInputNode(UserInputNode):
    def __init__(self, value:int=0, name:str="int_user_input"):
        self.value = int(value)
        super().__init__(name)

    def ui(self):
        self.input_number = ui.number(value=self.value, label=self.node_name).bind_value(self, 'value')

    def process(self) -> int:
        return int(self.value)
    

class FloatUserInputNode(UserInputNode):
    def __init__(self, value:float=0.0, name:str="float_user_input"):
        self.value = float(value)
        super().__init__(name)

    def ui(self):
        self.input_number = ui.number(value=self.value, label=self.node_name, format='%.2f').bind_value(self, 'value')

    def process(self) -> float:
        return float(self.value)
    

class StringUserInputNode(UserInputNode):
    def __init__(self, value:str="", name:str="string_user_input"):
        self.value = str(value)
        super().__init__(name)

    def ui(self):
        self.input_string = ui.input(value=self.value, label=self.node_name).bind_value(self, 'value')

    def process(self) -> str:
        return str(self.value)
    

class TextAreaInputNode(UserInputNode):
    def __init__(self, value:str="", name:str="text_area_user_input"):
        self.value = str(value)
        super().__init__(name)

    def ui(self):
        self.input_text = ui.textarea(value=self.value, label=self.node_name).bind_value(self, 'value')

    def process(self) -> str:
        return str(self.value)
    

class BoolUserInputNode(UserInputNode):
    def __init__(self, value:bool=False, name:str="bool_user_input"):
        self.value = bool(value)
        super().__init__(name)

    def ui(self):
        self.input_bool = ui.checkbox(value=self.value).bind_value(self, 'value')

    def process(self) -> bool:
        return bool(self.value)
    

class SizeUserInputNode(UserInputNode):
    def __init__(self, value:SizeType=(512, 512), name:str="size_user_input"):
        self.width = value[0]
        self.height = value[1]
        super().__init__(name)

    def ui(self):
        self.input_width = ui.number(value=self.width, label="Width").bind_value(self, 'width')
        self.input_height = ui.number(value=self.height, label="Height").bind_value(self, 'height')

    def process(self) -> SizeType:
        return (int(self.width), int(self.height))
    

class MinMaxIntInputNode(UserInputNode):
    def __init__(self, value:Tuple[int, int]=(0, 100), name:str="min_max_user_input"):
        self.min = value[0]
        self.max = value[1]
        super().__init__(name)

    def ui(self):
        self.input_min = ui.number(value=self.min, label="Min").bind_value(self, 'min')
        self.input_max = ui.number(value=self.max, label="Max").bind_value(self, 'max')

    def process(self) -> Tuple[int, int]:
        return (int(self.min), int(self.max))


class MinMaxFloatInputNode(UserInputNode):
    def __init__(self, value:Tuple[float, float]=(0.0, 100.0), name:str="min_max_float_user_input"):
        self.min = value[0]
        self.max = value[1]
        super().__init__(name)

    def ui(self):
        self.input_min = ui.number(value=self.min, label="Min", format='%.2f').bind_value(self, 'min')
        self.input_max = ui.number(value=self.max, label="Max", format='%.2f').bind_value(self, 'max')

    def process(self) -> Tuple[float, float]:
        return (float(self.min), float(self.max))
    

class BoolListUserInputNode(UserInputNode):
    def __init__(self, value:List[bool], labels:List[str], name:str="bool_list_user_input"):
        self.value:List[bool] = value
        self.labels:List[str] = labels
        super().__init__(name)

    def ui(self):
        for i in range(len(self.value)):
            self.checkbox(i)

    def checkbox(self, index:int):
        with ui.column().classes('gap-0'):
            ui.label(self.labels[index]).style('font-size: 12px; color:rgba(255, 255, 255, 0.7)')
            ui.checkbox(value=self.value[index], on_change=lambda e : self.update(index, e.value))

    def update(self, index:int, value:bool):
        self.value[index] = value

    def process(self) -> List[bool]:
        return self.value
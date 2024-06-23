from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from nicegui import ui
from enum import Enum, EnumType
import random


class UserInputNode(FunctionalNode):
    """ represents input from the user """

    def __init__(self, name:str="user_input"):
        super().__init__(name)
        self.update_listeners = []
        self.deepcopy_excluded_modules = ["nicegui"]

    def addUpdateListener(self, listener:Callable[[], None]):
        self.update_listeners.append(listener)

    def fireUpdate(self):
        for listener in self.update_listeners:
            listener()

    def getValue(self):
        raise Exception(f"{self.__class__.__name__} getValue() method is not implemented")
    
    def setValue(self, value):
        raise Exception(f"{self.__class__.__name__} setValue() method is not implemented")
    
    def gui(self):
        ui.label(self.node_name).classes('align-middle')


class IntUserInputNode(UserInputNode):
    def __init__(self, value:int|None=0, name:str="int_user_input"):
        self.value = int(value) if value is not None else None
        super().__init__(name)

    def getValue(self) -> int|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        self.input = ui.number(value=self.value, label=self.node_name).bind_value(self, 'value')

    def process(self) -> int|None:
        return int(self.value) if self.value is not None else None
    

class SeedUserInputNode(UserInputNode):
    MAX_SEED = 2**32 - 1

    def __init__(self, value:int|None=None, name:str="seed_user_input"):
        self.value = int(value) if value is not None else None
        super().__init__(name)

    def getValue(self) -> int|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        self.input = ui.number(value=self.value, label=self.node_name).bind_value(self, 'value')

    def process(self) -> int|None:
        if(self.value is None):
            return random.randint(0, self.MAX_SEED)
        else:
            return int(self.value)
    

class FloatUserInputNode(UserInputNode):
    def __init__(self, value:float=0.0, format:str='%.2f' , name:str="float_user_input"):
        self.value = float(value)
        self.format = format
        super().__init__(name)

    def getValue(self) -> float|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        ui.number(value=self.value, label=self.node_name, format=self.format).bind_value(self, 'value')

    def process(self) -> float:
        return float(self.value)
    

class BoolUserInputNode(UserInputNode):
    def __init__(self, value:bool=False, name:str="bool_user_input"):
        self.value = bool(value)
        super().__init__(name)

    def getValue(self) -> bool|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        ui.label(self.node_name).classes('align-middle')
        ui.checkbox(value=self.value).bind_value(self, 'value')

    def process(self) -> bool:
        return bool(self.value)
    

class StringUserInputNode(UserInputNode):
    def __init__(self, value:str="", name:str="string_user_input"):
        self.value = str(value)
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        ui.input(value=self.value, label=self.node_name).bind_value(self, 'value').classes('grow')

    def process(self) -> str:
        return str(self.value)
    

class TextAreaInputNode(UserInputNode):
    def __init__(self, value:str="", name:str="text_area_user_input"):
        self.value = str(value)
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.value
    
    def setValue(self, value):
        self.value = value

    def gui(self):
        ui.textarea(value=self.value, label=self.node_name).bind_value(self, 'value').classes('grow')

    def process(self) -> str:
        return str(self.value)
    

class ListSelectUserInputNode(UserInputNode):
    def __init__(self, value:str, options:List[str], name:str="list_select_user_input"):
        self.value = value
        self.options = sorted(options)
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.value
    
    def setValue(self, value):
        try:
            self.value = value
        except:
            pass

    def setOptions(self, options:List[str]):
        self.options = sorted(options)
        if(len(self.options) > 0):
            self.value = self.options[0]
        self.gui.refresh()

    @ui.refreshable
    def gui(self):
        ui.select(options=self.options, label=self.node_name).bind_value(self, 'value').classes('grow')

    def process(self) -> str:
        return str(self.value)
    

class DictSelectUserInputNode(UserInputNode):
    def __init__(self, value:Any, options:Dict[str, Any], name:str="dict_select_user_input"):
        self.value = value
        self.options = sorted(options.keys())
        self.selected = self.options[0]
        self.dict = options
        super().__init__(name)

    def getValue(self) -> str|None:
        return self.selected
    
    def setValue(self, value):
        try:
            self.selected = value
        except:
            pass

    def getSelectedOption(self) -> Any:
        return self.dict[self.selected]

    def setOptions(self, options:Dict[str, Any]):
        self.options = sorted(options.keys())
        self.dict = options
        if(len(self.options) > 0):
            self.value = self.options[0]
        self.gui.refresh()

    @ui.refreshable
    def gui(self):
        ui.select(options=self.options, label=self.node_name, on_change=self.fireUpdate).bind_value(self, 'selected').classes('grow')

    def process(self) -> Any:
        return self.dict[self.selected]


class EnumSelectUserInputNode(UserInputNode):
    def __init__(self, value:Any, enum:type[Enum], name:str="enum_select_user_input"):
        self.value = value.value
        self.enum = enum
        self.options = [option.value for option in self.enum]
        super().__init__(name)

    def getValue(self) -> str|None:
        return str(self.value)
    
    def setValue(self, value:str):
        self.value = str(value)

    def gui(self):
        ui.select(options=self.options, label=self.node_name).bind_value(self, 'value').classes('grow')

    def process(self) -> Enum|None:
        for member in self.enum:
            if member.value == self.value:
                return member
        return None


class SizeUserInputNode(UserInputNode):
    def __init__(self, value:SizeType=(512, 512), name:str="size_user_input"):
        self.width = value[0]
        self.height = value[1]
        super().__init__(name)

    def getValue(self) -> SizeType:
        return (self.width, self.height)
    
    def setValue(self, value:SizeType):
        self.width = value[0]
        self.height = value[1]

    def gui(self):
        ui.number(value=self.width, label="Width").bind_value(self, 'width')
        ui.button(icon = "swap_horiz", on_click=self.swap).props('dense').classes('align-middle')
        ui.number(value=self.height, label="Height").bind_value(self, 'height')

    def swap(self):
        self.width, self.height = self.height, self.width

    def process(self) -> SizeType:
        return (int(self.width), int(self.height))
    

class IntTupleInputNode(UserInputNode):
    def __init__(self, value:Tuple[int, int]=(0, 100), labels:Tuple[str, str]=("Min", "Max"), name:str="min_max_user_input"):
        self.value1 = value[0]
        self.value2 = value[1]
        self.labels = labels
        super().__init__(name)

    def getValue(self) -> Tuple[int, int]:
        return (self.value1, self.value2)
    
    def setValue(self, value:Tuple[int, int]):
        try:
            self.value1 = value[0]
            self.value2 = value[1]
        except:
            print(f"Invalid value for MinMaxFloatInputNode: {value}")

    def gui(self):
        ui.number(value=self.value1, label="Min").bind_value(self, 'value1')
        ui.number(value=self.value2, label="Max").bind_value(self, 'value2')

    def process(self) -> Tuple[int, int]:
        return (int(self.value1), int(self.value2))


class FloatTupleInputNode(UserInputNode):
    def __init__(self, value:Tuple[float, float]=(0.0, 100.0), labels:Tuple[str, str]=("Min", "Max"), name:str="float_tuple_user_input"):
        self.value1 = value[0]
        self.value2 = value[1]
        self.labels = labels
        super().__init__(name)

    def getValue(self) -> Tuple[float, float]:
        return (self.value1, self.value2)
    
    def setValue(self, value:Tuple[float, float]):
        try:
            self.value1 = value[0]
            self.value2 = value[1]
        except:
            print(f"Invalid value for MinMaxFloatInputNode: {value}")

    def gui(self):
        ui.number(value=self.value1, label=self.labels[0], format='%.2f').bind_value(self, 'value1')
        ui.number(value=self.value2, label=self.labels[1], format='%.2f').bind_value(self, 'value2')

    def process(self) -> Tuple[float, float]:
        return (float(self.value1), float(self.value2))
    

class BoolListUserInputNode(UserInputNode):
    def __init__(self, value:List[bool], labels:List[str], name:str="bool_list_user_input"):
        self.value:List[bool] = value
        self.labels:List[str] = labels
        super().__init__(name)

    def getValue(self) -> List[bool]:
        return self.value
    
    def setValue(self, value:List[bool]):
        self.value = value

    def gui(self):
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
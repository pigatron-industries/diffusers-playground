from diffuserslib.functional import *
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from .TrainLoraNode import *
from nicegui import ui


class TrainDataUserInputNode(UserInputNode):

    def __init__(self, name:str="train_data_input"):
        self.train_data:TrainDataType = [([""], 1)]
        super().__init__(name)


    def getValue(self) -> TrainDataType:
        return self.train_data
    

    def setValue(self, value:TrainDataType):
        try:
            self.train_data = value
        except:
            self.train_data = []


    @ui.refreshable
    def gui(self):
        with ui.row().classes('w-full'):
            ui.label().classes('w-8')
            ui.button(icon="add", on_click = lambda e: self.addInput(0)).props('dense').classes('align-middle')
        for i in range(len(self.train_data)):
            self.trainFilesGui(i)


    def trainFilesGui(self, i):
        train_files = self.train_data[i]
        with ui.row().classes('w-full'):
            ui.label().classes('w-8')
            ui.input(value=train_files[0][0], label="Train Files", on_change=lambda e: self.updateTrainFiles(i, e.value)).classes('grow')
            ui.number(value=train_files[1], label="Repeats", on_change=lambda e: self.updateRepeats(i, e.value)).classes('small-number')
            ui.button(icon="remove", on_click = lambda e: self.removeInput(i)).props('dense').classes('align-middle')


    def addInput(self, i):
        self.train_data.insert(i, ([""], 1))
        self.gui.refresh()

    
    def removeInput(self, i):
        self.train_data.pop(i)
        self.gui.refresh()


    def updateTrainFiles(self, i, value):
        train_files = self.train_data[i]
        self.train_data[i] = ([value], train_files[1])

        
    def updateRepeats(self, i, value):
        train_files = self.train_data[i]
        self.train_data[i] = (train_files[0], int(value))


    def process(self) -> TrainDataType:
        return self.train_data
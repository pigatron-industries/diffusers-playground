from nicegui import ui
from typing import List, Callable
from diffuserslib.GlobalConfig import GlobalConfig
import os


class FileDialog():

    def __init__(self, callback:Callable[[List[str]], None]):
        self.callback = callback
        self.selected = []
        self.createDialog()


    @ui.refreshable
    def createDialog(self):
        with ui.dialog() as dialog:
            with ui.card():
                ui.tree(self.getFileTree(), label_key='id', tick_strategy='leaf', on_tick=lambda e: self.selectFile(e.value))
                ui.button('Done', on_click=self.close)
        self.dialog = dialog


    def selectFile(self, values):
        self.selected = values


    def getFileTree(self):
        paths = GlobalConfig.inputs_dirs
        tree = []
        for path in paths:
            tree.append({'id': path, 'children': self.getDirTree(path)})
        print(tree)
        return tree
            

    def getDirTree(self, path:str):
        tree = []
        for subpath in os.listdir(path):
            fullpath = os.path.join(path, subpath)
            if os.path.isdir(fullpath):
                tree.append({'id': fullpath, 'children': self.getDirTree(fullpath)})
            else:
                tree.append({'id': fullpath})
        return tree


    def open(self):
        self.createDialog.refresh()
        self.dialog.open()


    def close(self):
        self.dialog.close()
        self.callback(self.selected)
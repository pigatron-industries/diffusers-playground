from nicegui import ui, run
from typing import List, Callable
from diffuserslib.GlobalConfig import GlobalConfig
import os


class FileDialog():

    def __init__(self, callback:Callable[[List[str]], None], file_extensions:List[str]):
        self.callback = callback
        self.file_extensions = file_extensions
        self.selected = []
        self.filetree = []
        self.createDialog() # type: ignore


    @ui.refreshable
    def createDialog(self):
        with ui.dialog() as dialog:
            with ui.card():
                self.tree = ui.tree(self.filetree, label_key='id', tick_strategy='leaf', on_tick=lambda e: self.selectFile(e.value))
                ui.button('Done', on_click=self.close)
        self.dialog = dialog


    def selectFile(self, values):
        self.selected = values


    def getFileTree(self):
        print("Loading input file tree...")
        paths = GlobalConfig.inputs_dirs
        tree = []
        for path in paths:
            if (os.path.exists(path)):
                tree.append({'id': path, 'children': self.getDirTree(path)})
        print("Input file tree loaded")
        return tree
            

    def getDirTree(self, path:str):
        tree = []
        for subpath in os.listdir(path):
            fullpath = os.path.join(path, subpath)
            if (os.path.isdir(fullpath)):
                tree.append({'id': fullpath, 'children': self.getDirTree(fullpath)})
            elif (os.path.splitext(fullpath)[1][1:] in self.file_extensions):
                    tree.append({'id': fullpath})
        return tree


    async def open(self):
        self.filetree = await run.io_bound(self.getFileTree)
        self.createDialog.refresh()
        self.dialog.open()


    def close(self):
        self.dialog.close()
        self.callback(self.selected)
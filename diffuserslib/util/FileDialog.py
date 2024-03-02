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
                self.tree = ui.tree(self.filetree, tick_strategy='strict', on_tick=lambda e: self.selectFile(e.value), on_expand=lambda e: self.onExpand(e.value))
                ui.button('Done', on_click=self.close)
        self.dialog = dialog


    async def onExpand(self, expanded_paths:List[str]):
        def expand():
            for path in expanded_paths:
                print(path)
                self.populateGrandChildren(path)
        await run.io_bound(expand)


    def selectFile(self, values):
        self.selected = values


    def initFileTree(self):
        self.filetree = []
        paths = GlobalConfig.inputs_dirs
        for path in paths:
            if (os.path.exists(path)):
                node = {'id': path, 'label': path, 'children': []}
                self.populateChildren(node)
                self.filetree.append(node)
                


    def populateChildren(self, node):
        path = node['id']
        if(len(node['children']) == 0 and os.path.isdir(path)):
            dirlist = os.listdir(path)
            # dirlist = await run.io_bound(os.listdir, path)
            for subpath in dirlist:
                fullpath = os.path.join(path, subpath)
                if (os.path.isdir(fullpath)):
                    node['children'].append({'id': fullpath, 'label': subpath, 'children': []})
                elif (os.path.splitext(fullpath)[1][1:] in self.file_extensions):
                    node['children'].append({'id': fullpath, 'label': subpath, 'children': []})


    def populateGrandChildren(self, path:str):
        node = self.findNode(path, self.filetree)
        if (node is not None):
            for child in node['children']:
                self.populateChildren(child)


    def findNode(self, path:str, tree:List[dict]):
        for node in tree:
            if (node['id'] == path):
                return node
            else:
                result = self.findNode(path, node['children'])
                if (result is not None):
                    return result
        return None


    async def open(self):
        await run.io_bound(self.initFileTree)
        self.createDialog.refresh()
        self.dialog.open()


    def close(self):
        self.dialog.close()
        self.callback(self.selected)
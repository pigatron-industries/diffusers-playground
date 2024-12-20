import platform
from pathlib import Path
from typing import Optional

from nicegui import events, ui
from typing import List


class LocalFilePicker(ui.dialog):

    def __init__(self, directory: str, submit_text:str = "Ok", upper_limit: Optional[str] = ..., 
                 drives: List[str]|None = None, multiple: bool = False, show_hidden_files: bool = False) -> None:
        """Local File Picker

        This is a simple file picker that allows you to select a file from the local filesystem where NiceGUI is running.

        :param directory: The directory to start in.
        :param upper_limit: The directory to stop at (None: no limit, default: same as the starting directory).
        :param multiple: Whether to allow multiple files to be selected.
        :param show_hidden_files: Whether to show hidden files.
        """
        super().__init__()

        self.multiple = multiple
        self.drives = drives
        self.path = Path(directory).expanduser()
        self.filename = None
        if upper_limit is None:
            self.upper_limit = None
        else:
            self.upper_limit = Path(directory if upper_limit == ... else upper_limit).expanduser()
        self.show_hidden_files = show_hidden_files

        with self, ui.card():
            self.add_drives_toggle()
            if(not multiple):
                ui.input('File').bind_value(self, 'filename').classes('w-full')
            self.grid = ui.aggrid({
                'columnDefs': [{'field': 'name', 'headerName': 'File'}],
                'rowSelection': 'multiple' if multiple else 'single',
            }, html_columns=[0]).classes('w-96').on('cellDoubleClicked', self.handle_double_click).on('cellClicked', self.handle_click)
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=self.close).props('outline')
                ui.button(submit_text, on_click=self._handle_ok)
        self.update_grid()

    def add_drives_toggle(self):
        # if platform.system() == 'Windows':
            # import win32api
            # drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
        if(self.drives is not None):
            self.drives_toggle = ui.toggle(self.drives, value=self.drives[0], on_change=self.update_drive)

    def update_drive(self):
        self.path = Path(self.drives_toggle.value).expanduser()
        self.update_grid()

    def update_grid(self) -> None:
        paths = list(self.path.glob('*'))
        if not self.show_hidden_files:
            paths = [p for p in paths if not p.name.startswith('.')]
        paths.sort(key=lambda p: p.name.lower())
        paths.sort(key=lambda p: not p.is_dir())

        self.grid.options['rowData'] = [
            {
                'name': f'📁 <strong>{p.name}</strong>' if p.is_dir() else p.name,
                'path': str(p),
            }
            for p in paths
        ]
        if self.upper_limit is None and self.path != self.path.parent or \
                self.upper_limit is not None and self.path != self.upper_limit:
            self.grid.options['rowData'].insert(0, {
                'name': '📁 <strong>..</strong>',
                'path': str(self.path.parent),
            })
        self.grid.update()

    def handle_click(self, e: events.GenericEventArguments) -> None:
        selected_path = Path(e.args['data']['path'])
        self.filename = selected_path.name

    def handle_double_click(self, e: events.GenericEventArguments) -> None:
        self.path = Path(e.args['data']['path'])
        if self.path.is_dir():
            self.update_grid()
            self.filename = None
        else:
            self.submit([str(self.path)])

    async def _handle_ok(self):
        rows = await ui.run_javascript(f'getElement({self.grid.id}).gridOptions.api.getSelectedRows()')
        if(not self.multiple and self.filename is not None):
            self.submit([f"{self.path}/{self.filename}"])
        else:
            self.submit([r['path'] for r in rows])

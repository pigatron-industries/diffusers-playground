from nicegui import ui, context
from .api import *


class AbstractView:

    splitter_position = 40

    def page(self):
        ui.page_title('Generative Toolkit')
        ui.label.default_classes('label')
        ui.select.default_classes('w-80')
        ui.query('body').style(f'background-color: rgb(24, 28, 33)')
        ui.add_head_html('''
            <style>
                .align-middle {
                    transform: translateY(50%);
                }
                .small-number {
                    width: 4em;
                }
                .h-full-input .q-field__control {
                    height:100%;
                }
            </style>
        ''')
        context.get_client().content.classes('h-[100vh] p-0')

        with ui.column().classes("w-full h-full no-wrap gap-0"):
            with ui.row().classes("w-full p-2 place-content-between").style("background-color:#2b323b; border-bottom:1px solid #585b5f"):
                self.toggles()
                self.status()
                with ui.row():
                    self.buttons()
                    ui.button(icon='settings', color='dark', on_click=self.settings)
                    
            with ui.splitter(value=self.splitter_position).classes("w-full h-full no-wrap overflow-auto") as splitter:
                with splitter.before:
                    with ui.column().classes("p-2 w-full"):
                        self.controls()
                with splitter.after:
                    self.outputs()

    def toggles(self):
        pass

    def status(self):
        pass

    def buttons(self):
        pass

    def settings(self):
        pass

    def controls(self):
        pass

    def outputs(self):
        pass
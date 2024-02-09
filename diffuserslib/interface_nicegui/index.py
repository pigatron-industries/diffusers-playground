from nicegui import ui, context
from .api import *





class Model:
    def __init__(self):
        self.workflows = loadWorkflows()


@ui.page('/')
def gui():
    model = Model()
    context.get_client().content.classes('h-[100vh]')
    with ui.column().classes("w-full h-full no-wrap").style("height: 100%"):
        with ui.splitter().classes("w-full h-full no-wrap").style("height: 100%") as splitter:
            with splitter.before:
                ui.label('Workflow').classes('mr-2')
                workflow = ui.select(list(model.workflows.keys()))

                # for widget in model.widgets:
                #     with ui.row():
                #         if widget == 'input':
                #             ui.label(widget).classes('mr-2')
                #             ui.input()
                #         else:
                #             ui.label(widget).classes('mr-2')
                #             ui.select(["test1", "test2", "test3"])

            with splitter.after:
                ui.label('This is some content on the right hand side.').classes('ml-2')




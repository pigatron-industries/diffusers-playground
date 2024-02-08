from nicegui import ui, app

@ui.page('/')
def gui():
    with ui.splitter() as splitter:
        with splitter.before:
            ui.label('This is some content on the left hand side.').classes('mr-2')
        with splitter.after:
            ui.label('This is some content on the right hand side.').classes('ml-2')




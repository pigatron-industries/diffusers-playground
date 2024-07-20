from llama_index.core.llms import ChatMessage, MessageRole
from nicegui import ui


class ChatHistoryMessageControls:
    def __init__(self, id:int, message:ChatMessage|None):
        self.id = id
        self.message = message
        self.markdown_control = None
        self.gui() 


    def gui(self):
        role = self.message.role if self.message is not None else MessageRole.ASSISTANT
        color = "#264927" if role == MessageRole.ASSISTANT else "#2E4053"
        text = self.message.content if self.message is not None else ""
        with ui.card_section().classes('w-full').style(f"background-color:{color}; border-radius:8px;"):
            with ui.row().classes('grow'):
                with ui.column().classes('grow'):
                    self.markdown_control = ui.markdown(text)


    def update(self, message:ChatMessage):
        assert self.markdown_control is not None
        self.message = message
        text = self.message.content if self.message is not None else ""
        self.markdown_control.set_content(text)
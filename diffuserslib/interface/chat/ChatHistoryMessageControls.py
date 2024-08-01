from llama_index.core.llms import ChatMessage, MessageRole
from nicegui import ui
from diffuserslib.interface.chat.ChatController import ChatController


class ChatHistoryMessageControls:
    def __init__(self, id:int, message:ChatMessage|None, controller:ChatController):
        self.id = id
        self.message = message
        self.controller = controller
        self.output_control = None
        self.gui() 


    def gui(self):
        role = self.message.role if self.message is not None else MessageRole.ASSISTANT
        color = "#264927" if role == MessageRole.ASSISTANT else "#2E4053"
        text = self.getMarkdown()
        with ui.card_section().classes('w-full').style(f"background-color:{color}; border-radius:8px;") as self.container:
            with ui.row().classes('grow no-wrap'):
                with ui.column().classes('grow'):
                    self.output_control = ui.markdown(text)
                with ui.column():
                    ui.button(icon='delete', on_click=self.deleteMessage).props('dense')


    def getMarkdown(self):
        if self.message is None or self.message.content is None:
            return ""
        text = self.message.content.lstrip()
        lines = text.split('\n')
        output = ""
        for line in lines:
            if line.lstrip().startswith("```"):
                line = line.lstrip()
            output += f"{line}\n"
        return output


    def deleteMessage(self):
        self.controller.removeMessage(self.id)
        self.container.delete()


    def update(self, message:ChatMessage):
        assert self.output_control is not None
        self.message = message
        text = self.getMarkdown()
        self.output_control.set_content(text)
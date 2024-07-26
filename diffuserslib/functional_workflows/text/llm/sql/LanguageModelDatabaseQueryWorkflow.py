from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text import TemplateNode, ExtractTextNode
from diffuserslib.functional.nodes.text.db import DatabaseMetadataNode, DatabaseQueryNode, DatabaseConnections
from diffuserslib.functional.nodes.text.llm.OllamaModels import OllamaModels
from diffuserslib.functional.nodes.text.llm.LanguageModelCompletionNode import LanguageModelCompletionNode
from diffuserslib.functional_workflows.text.llm.sql.DatabasePrompts import DatabasePrompts


class LanguageModelDatabaseQueryWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Language Model Database Query", str, workflow=True, converse=True)


    def build(self):
        models = [""]
        try:
            models = list(OllamaModels.loadLocalModels().keys())
        except:
            print("Error loading Ollama models. Is Ollama running?")
            pass

        db_connections = DatabaseConnections.getDatabaseConnections()
        db_connections_dict = {connection.name: connection for connection in db_connections.values()}
        
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model")
        db_connection_input = DictSelectUserInputNode(value = "", options = db_connections_dict, name = "db_connection")
        message_input = TextAreaInputNode(value = "", name = "message_input")

        table_info = DatabaseMetadataNode(connection=db_connection_input, name="db_metadata")
        create_sql_prompt = TemplateNode(DatabasePrompts.CREATE_SQL_TEMPLATE, tableinfo=table_info, question=message_input, name="create_sql_prompt")
        sql_query = LanguageModelCompletionNode(model=model_input, prompt=create_sql_prompt, name="llm")
        extract_sql = ExtractTextNode(text=sql_query, start_token="```sql", end_token="```", name="extract_sql")
        sql_result = DatabaseQueryNode(connection=db_connection_input, query=extract_sql, output_type="markdown", name="db_query")
        final_output_prompt = TemplateNode(DatabasePrompts.FINAL_OUTPUT_TEMPLATE, question=message_input, tableinfo=table_info, sql_query=sql_query, sql_result=sql_result, name="final_output_prompt")
        final_output = LanguageModelCompletionNode(model=model_input, prompt=final_output_prompt, name="llm")
        return final_output
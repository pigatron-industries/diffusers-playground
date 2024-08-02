from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.control import *
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
        
        model_input = ListSelectUserInputNode(value = "llama3:8b", options = models, name = "model_input")
        db_connection_input = DictSelectUserInputNode(value = "", options = db_connections_dict, name = "db_connection")
        message_input = TextAreaInputNode(value = "", name = "message_input")
        runquery_input = BoolUserInputNode(value = False, name = "run_query")
        
        table_info = DatabaseMetadataNode(connection=db_connection_input, name="db_metadata")
        create_sql_prompt = TemplateNode(DatabasePrompts.CREATE_SQL_TEMPLATE, tableinfo=table_info, question=message_input, name="create_sql_prompt")
        sql_query = LanguageModelCompletionNode(model=model_input, prompt=create_sql_prompt, name="llm_query").setDisplayIndex(0)

        extract_sql = ExtractTextNode(text=sql_query, start_token="```sql", end_token="```", name="extract_sql")
        sql_result = DatabaseQueryNode(connection=db_connection_input, query=extract_sql, output_type="markdown", name="db_query").setDisplayIndex(1)
        final_output_prompt = TemplateNode(DatabasePrompts.FINAL_OUTPUT_TEMPLATE, question=message_input, tableinfo=table_info, sql_query=extract_sql, sql_result=sql_result, name="final_output_prompt")
        final_output = LanguageModelCompletionNode(model=model_input, prompt=final_output_prompt, name="llm_result").setDisplayIndex(2)

        # return final_output

        run_query_conditional = ConditionalNode(condition=runquery_input, false=sql_query, true=final_output, name="run_query_conditional")
        return run_query_conditional
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .DatabaseConnections import DatabaseConnection
from typing import Sequence


class DatabaseQueryNode(FunctionalNode):
    def __init__(self, 
                 connection:StringFuncType,
                 query:StringFuncType,
                 output_type:str = "list",
                 name:str = "db_query"):
        super().__init__(name)
        self.addParam("connection", connection, str)
        self.addParam("query", query, str)
        self.output_type = output_type


    def process(self, connection:str|DatabaseConnection, query:str) -> str|Sequence:
        from sqlalchemy import create_engine, sql
        from sqlalchemy.orm import Session

        if(query is None or query == ""):
            return "No query executed"

        if isinstance(connection, DatabaseConnection):
            connectionstring = connection.connection
        else:
            connectionstring = connection

        engine = create_engine(connectionstring, echo=True)
        session = Session(engine)

        # remove any trailing semicolons
        query = query.strip().rstrip(";")

        try:
            statement = sql.text(query)
            result = session.execute(statement)
            if(self.output_type == "markdown"):
                # print("DatabaseQueryNode: ", result)
                return self.toMarkdown(result)
            else:
                return result.fetchall()
        except Exception as e:
            session.rollback()
            return f"Error: {e}"
        finally:
            session.close()
            engine.dispose()

        
    def toMarkdown(self, result) -> str:
        columns = result.keys()
        markdown_table = "| " + " | ".join(columns) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        for row in result:
            markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
        return markdown_table
        

    def progress(self) -> WorkflowProgress|None:
        if(self.output is None):
            return WorkflowProgress(0, None)
        else:
            return WorkflowProgress(1, self.output)

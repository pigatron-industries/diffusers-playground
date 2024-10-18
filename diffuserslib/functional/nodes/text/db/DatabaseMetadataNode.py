from torch import le
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .DatabaseConnections import DatabaseConnection
from typing import List


class DatabaseMetadataNode(FunctionalNode):
    def __init__(self, 
                 connection:StringFuncType,
                 exclude_tables:List[str] = [],
                 include_tables:List[str] = [],
                 cache:bool = True,
                 name:str = "db_query"):
        super().__init__(name)
        self.addParam("connection", connection, str)
        self.addParam("exclude_tables", exclude_tables, List[str])
        self.addParam("include_tables", include_tables, List[str])
        self.cache = cache
        self.create_statements = None


    def process(self, connection:str|DatabaseConnection, exclude_tables:List[str], include_tables:List[str]) -> str:
        from sqlalchemy import create_engine
        if(self.cache and self.create_statements is not None):
            print("DatabaseMetadataNode: using cache")
            return self.create_statements

        if isinstance(connection, DatabaseConnection):
            connectionstring = connection.connection
            exclude_tables = connection.exclude_tables
            include_tables = connection.include_tables
        else:
            connectionstring = connection

        engine = create_engine(connectionstring, echo=True)

        create_statements = ""
        if(len(include_tables)>0):
            schema_dict = {}
            for tablename in include_tables:
                schemaname = None
                if("." in tablename):
                    schemaname = tablename.split(".")[0]
                    tablename = tablename.split(".")[1]
                if schemaname not in schema_dict:
                    schema_dict[schemaname] = []
                schema_dict[schemaname].append(tablename)
            for schemaname in schema_dict:
                create_statements += self.getMetadata(engine, schemaname, exclude_tables, schema_dict[schemaname])
        else:
            create_statements += self.getMetadata(engine, None, exclude_tables, include_tables)

        print("DatabaseMetadataNode: ", create_statements)
        self.create_statements = create_statements
        return create_statements
        

    def getMetadata(self, engine, schemaname, exclude_tables, include_tables):
        from sqlalchemy import MetaData
        metadata = MetaData()
        metadata.reflect(bind=engine, schema=schemaname, only=include_tables)

        create_statements = ""
        for table_name, table in metadata.tables.items():
            if len(include_tables)>0 and table_name.lower() not in (schemaname.lower() + "." + name.lower() for name in include_tables):
                continue
            if table_name.lower() in (name.lower() for name in exclude_tables):
                continue
            create_statement = f"CREATE TABLE {table_name} (\n"
            # Iterate over the columns
            columns = []
            for column in table.columns:
                column_def = f"    {column.name} {column.type}"
                if column.default is not None:
                    column_def += f" DEFAULT {column.default.arg}"
                if not column.nullable:
                    column_def += " NOT NULL"
                columns.append(column_def)
            # Add primary key constraint
            if table.primary_key:
                pk_columns = ", ".join([col.name for col in table.primary_key.columns])
                columns.append(f"    PRIMARY KEY ({pk_columns})")
            # Add foreign key constraints
            for fk in table.foreign_keys:
                fk_def = f"    FOREIGN KEY({fk.parent.name}) REFERENCES {fk.column.table.name} ({fk.column.name})"
                columns.append(fk_def)
            create_statement += ",\n".join(columns)
            create_statement += "\n);"
            create_statements += create_statement + "\n\n"
        return create_statements
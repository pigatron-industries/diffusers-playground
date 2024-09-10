

class DatabasePrompts:

    POSTRESQL_SPECIFIC = """
Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause.
    """

    ORACLE_SPECIFC = """
Pay attention to use SYSDATE function to get the current date, if the question involves "today".
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the ROWNUM clause.
"""

    DBMS = {
        "PostgreSQL": {
            "dbmsinfo": POSTRESQL_SPECIFIC,
        }, 
        "Oracle": {
            "dbmsinfo": ORACLE_SPECIFC,
        }
    }

    SHARED_TEMPLATE = """
You are a {dbms} expert. Given an input question, first create a syntactically correct {dbms} query to run, then look at the results of the query and return the answer to the input question. 
You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
{dbmsinfo}

Only use the following tables:
```
{tableinfo}
```

{extrainfo}

"""


    CREATE_SQL_TEMPLATE = SHARED_TEMPLATE + """
Use the following format:

Question: Question here

SQLQuery: 
```sql
SQL Query to run
```

Question: {question}
    """


    FINAL_OUTPUT_TEMPLATE = SHARED_TEMPLATE + """
Use the following format:

Question: Question here

SQLQuery: 
```sql
SQL Query to run
```

SQLResult: 
Result of the SQLQuery

Answer: Final answer here



Question: {question}

SQLQuery: 
```sql
{sql_query}
```

SQLResult:
{sql_result}

Answer:
    """

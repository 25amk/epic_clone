"""
src/site/ornl/frontier/db_test.py - Frontier (ORNL) system specific data access methods
"""
import json, re
from pathlib import Path
from textwrap import dedent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from src.config import get_settings
from src.tools.sql import JSONSQLDatabase
from pydantic import TypeAdapter, ImportString
from langchain_core.tools import tool
import pandas as pd

# Load settings
settings = get_settings()


def make_table_description(
    table_name: str, metadata_file: str | Path,
    relationships: list[str | None] = [],
) -> str:
    """
    Read from one of the metadata CSV files and create a table deescription

    The metadata file should look like below

    {
        'Column': 'allocation_id'
        'Description': 'Allocation ID is the unique job id or workload id.',
        'Type': 'string',
        'Anonymized': 'false',
        'isPrimary': 'true',
        'isForeign': 'false',
    }

    Returns a formatted string that looks like below
    '''
    "CREATE TABLE users ( \nuser PRIMARY VARCHAR(50) COMMENT 'It is the unique 
                          code of the users who submit the jobs/workloads on the system'
                          \ndescription VARCHAR(50) COMMENT 'Description of the user'\n)"
    '''
    """
    # Metadata file location is deduced from the table name

    df = pd.read_csv(metadata_file)
    create_schema_string = f"CREATE TABLE {table_name} ( \n"
    for row_ in df.iterrows():
        create_schema_string = create_schema_string+f"{row_[1]['Column']}"
        if row_[1]['isPrimary'] == 'true' or row_[1]['isPrimary'] == True:
            create_schema_string = create_schema_string + f" PRIMARY"

        if row_[1]['Type'] == 'string':
            create_schema_string = create_schema_string + f" VARCHAR(50) "
        elif row_[1]['Type'] == 'int' or row_[1]['Type'] == 'number':
            create_schema_string = create_schema_string + f" INT "
        elif row_[1]['Type'] == 'float':
            create_schema_string = create_schema_string + f" DOUBLE PRECISION "
        elif row_[1]['Type'] == 'timestamp':
            create_schema_string = create_schema_string + f" TIMESTAMP "
        elif row_[1]['Type'] == 'date':
            create_schema_string = create_schema_string + f" DATE "    

        if row_[1]['isForeign'] != 'false' and row_[1]['isForeign'] != False :
            #print('Foreign Key: ', row_[1]['isForeign'])
            create_schema_string = create_schema_string + f", {row_[1]['isForeign']} "

        create_schema_string = create_schema_string+ f"COMMENT '{row_[1]['Description']}'\n"
    create_schema_string = create_schema_string+")"
    return create_schema_string


def get_db(db_module: str) -> JSONSQLDatabase:
    db_module_obj = TypeAdapter(ImportString).validate_strings(db_module)
    return db_module_obj.create_sql_database()


def de_markdown(text):
    """Remove the unwanted markdown ticks the model injects"""
    match = re.fullmatch(r"(.*```.*?\n)?(.+)```.*", text, flags=re.DOTALL)
    if match:
        text = match[2]
    return text.strip()


def create_sql_qna_chain(model: BaseChatModel, db_module: str):
    """
    Factory for the sqldatabase_chain object
    Pass the sql model to use, and the db module to use. The db module should be a string name of
    a module, e.g. "src.site.ornl.frontier.db" which contains two functions `create_sql_database()`
    and `create_sql_prompt_template()`
    """
    db_module_obj = TypeAdapter(ImportString).validate_strings(db_module)
    db: JSONSQLDatabase = db_module_obj.create_sql_database()
    sql_prompt_template = db_module_obj.create_sql_prompt_template()

    execute_query = QuerySQLDatabaseTool(db=db)
    def agent_loop(arg):
        question = arg['question']
        generated_sql = None
        query_result = None
        error = None
        depth = 0
        MAX_DEPTH = 3
        messages = []

        while depth < MAX_DEPTH and query_result is None:
            prompt = sql_prompt_template.invoke({
                "question": question,
                "conversation": messages,
            })

            ai_msg = model.invoke(prompt)
            generated_sql = de_markdown(ai_msg.content)
            execute_result = execute_query.invoke(generated_sql)
            try:
                query_result = json.loads(execute_result)
                error = None
            except ValueError:
                query_result = None
                error = str(execute_result)
            messages.append(AIMessage(generated_sql))
            if error:
                error_message = dedent("""
                    The previous query failed with the following error:
                    {error}

                    Correct the SQL query.
                """).strip()
                messages.append(HumanMessage(error_message.format(error = error)))

            depth += 1

        return {
            **arg,
            "query": generated_sql,
            "query_result": query_result,
            "error": error,
        }

    return RunnableLambda(agent_loop)


def create_sql_qna_chain_as_tool(model: BaseChatModel, db_module: str):
    """
    Creates the sql_qna_chain as a langchain tool
    """
    chain = create_sql_qna_chain(model, db_module)

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def sql_qna_chain(question: str):
        """
        Use this tool to answer questions about tabular and time series data in the HPC telemetry database.

        Args:
            question: Natural language query

        Returns:
            The results of the query from the database
        """
        result = chain.invoke({"question": question})
        if result.get("error"):
            return result, []
        else:
            artifact = result
            message = (
                json.dumps({
                    **result,
                    "query_result": result['query_result'][:settings.query_output_limit_chat_model],
                }, indent = 4) +
                "\n" +
                "The user has been shown a table with the query result."
            )
            return message, artifact

    return sql_qna_chain

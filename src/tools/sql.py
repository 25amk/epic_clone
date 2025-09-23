"""
Some helper tools for SQL
"""
from typing import Any, override
import json, itertools
from pydantic import TypeAdapter
from langchain_community.utilities.sql_database import SQLDatabase, truncate_word

class JSONSQLDatabase(SQLDatabase):
    """
    langchain_community SQLDatabase returns its result as a string, and it just does a naive Python
    object stringification. E.g. it will output a string like
    ```
    [('3941042', datetime.datetime(2024, 1, 2, 14, 3, 50, tzinfo=<DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>))]
    ```
    
    This makes parsing the result string difficult, especially for things like datetimes. It also
    gives no information about the column names. This overrides SQLDatabase to return JSON results,
    with ISO formatted dates. 
    """
    def __init__(self, *args, max_result_length = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_result_length = max_result_length


    @override
    def run(self, command, fetch = "all", include_columns = True, *,
        parameters = None,
        execution_options = None,
    ):
        """ Override this to output JSON and allways show the columns. """
        cursor = self._execute(
            command, fetch, parameters=parameters, execution_options=execution_options
        )

        if fetch == "cursor":
            return cursor

        results = []
        # limit results returned
        for row in itertools.islice(cursor, 0, self.max_result_length):
            results.append({
                column: truncate_word(value, length=self._max_string_length)
                for column, value in row.items()
            })

        # We ignore the include_columns parameter
        ret = TypeAdapter(Any).serializer.to_python(results, mode='json')
        # Jsonize with newlines between each row
        ret = '[\n' + ',\n'.join(f'    {json.dumps(o)}' for o in ret) + "\n]"
        return ret

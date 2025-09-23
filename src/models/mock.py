from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

DEFAULT_RESPONSES = {
    'predict': AIMessage('',
        tool_calls=[{
            'name': 'job_pred',
            'args': {
                'question': "What temperature range should I expect for ocean current modeling on 11 nodes for 420 minutes?",
            },
            'id': '0',
        }],
    ),
    'sql': AIMessage('',
        tool_calls=[{
            'name': 'sql_qna_chain',
            'args': {
                'question': "Generate a report of total node hours for jobs by project_code, sorted by SCHEDULING POLICY and calculated for each day.",
            },
            'id': '0',
        }],
    ),
}

def mock_model(responses: dict[str,AIMessage]|None = None):
    """ Mock "top level" model for testing """
    if not responses:
        responses = DEFAULT_RESPONSES

    def func(msgs):
        msg = msgs[-1] if isinstance(msgs, list) else msgs
        logger.debug(f"CHAT: {msg}")

        if isinstance(msg, HumanMessage):
            key = msg.content.strip().lower()
            if key in responses:
                return responses[key]
            else:
                command_list = ", ".join(DEFAULT_RESPONSES.keys())
                return AIMessage(f"I don't know how to answer that. Known commands: {command_list}")
        else:
            return AIMessage("Tools called")

    model = RunnableLambda(func)
    model.bind_tools = lambda *args, **kwargs: model
    return model

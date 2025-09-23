import functools
from src.chains.chains import create_chat_chain_with_tool_router
from src.models import get_large_model, get_small_model_opt, get_sql_model_opt
from src.config import get_settings
settings = get_settings()

@functools.cache
def get_chat_chain():
    """
    Gets the chat chain and initializes it.
    Cached, so you can call it multiple times and get the same chain instance.
    This needs to be in a separate module from chatui.py since chainlit re-imports
    the chatui module.
    """
    large_model = get_large_model()
    small_model = get_small_model_opt()
    sql_model = get_sql_model_opt()
    runnable = create_chat_chain_with_tool_router(
        large_model = large_model,
        small_model = small_model,
        sql_model = sql_model,
        db_module = settings.db_module,
    )
    return runnable

from langchain_ollama import ChatOllama

def get_ollama_model(model_name: str, **kwargs):
    return ChatOllama(
        model=model_name,
        **kwargs,
    )

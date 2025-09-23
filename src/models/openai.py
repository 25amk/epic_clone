"""
src/models/openai.py: OpenAI model factories
"""
import urllib.parse
from langchain_openai import ChatOpenAI

def get_openai_model(
    name: str,
    url: str,
    key: str|None,
    **kwargs,
):
    """ Gets a model using an openai-compatible REST API """
    client_args = dict(
        model = name,
        api_key = key,
        stream_usage = True,
        temperature = 0,
    )

    if url != "default":
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path.removesuffix('/').removesuffix('/chat/completions')
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{path}"
        query = urllib.parse.parse_qs(parsed_url.query)
        query = {k: v[0] for k, v in query.items()}
        client_args.update(dict(
            base_url = base_url,
            default_query = query,
        ))

    client_args.update(**kwargs)
    return ChatOpenAI(**client_args)

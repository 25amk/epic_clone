"""
RAG (Retrieval Augmented Generation) implementation using ChromaDB for text, image, and table databases.
"""
import os
import torch
import pandas as pd
from typing import Any, List
from loguru import logger
from openai import OpenAI
from chromadb import PersistentClient
from docutils.core import publish_doctree
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import MarkdownTextSplitter
from tqdm.autonotebook import tqdm
from langchain_core.language_models.chat_models import BaseChatModel

from .model_cache import get_embedding_function
from .model_cache import get_llama_generator
from .model_cache import get_sentence_transformer
from .mxbai import MxbaiRetriever
from .mxbai import MxbaiReranker


class TextDatabase:
    """
    A class to handle text data storage and retrieval using ChromaDB.
    """
    def __init__(self, collection, model: BaseChatModel):
        self.collection = collection
        self.model = model
        logger.info("Initializing text database")
        logger.info("Loading Sentence Transformer model")
        self.text_model = get_sentence_transformer("mixedbread-ai/mxbai-embed-large-v1")
        logger.info("Loading MarkdownTextSplitter")
        self.splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
        logger.info("Loading MxbaiRetriever")
        self.retriever = MxbaiRetriever('mixedbread-ai/mxbai-embed-large-v1',collection)
        logger.info("Loading MxbaiRanker")
        self.ranker = MxbaiReranker('mixedbread-ai/mxbai-rerank-large-v1')

    def index(self, text):
        chunks = self.splitter.split_text(text)
        vectors = [self.text_model.encode(chunk).tolist() for chunk in chunks]
        metadata = [{"chunk": chunk} for chunk in chunks]
        ids = [str(len(self.collection.get()["ids"]) + i + 1) for i in range(len(chunks))]
        self.collection.add(embeddings=vectors, metadatas=metadata, ids=ids)

    def answer_query(self, query):
        query_vector = self.text_model.encode(query).tolist()
        result = self.collection.query(query_embeddings=[query_vector], n_results=9)
        retrieved_chunks = list(set([chunk_dict['chunk'] for chunk_dict in result['metadatas'][0]]))
        ranked_result = self.ranker.rank(query, retrieved_chunks, topk=3)

        question = ["Use the contexts provided below and answer the question following the contexts. The answer should be generated using the contexts only. If the contexts seems insufficient to answer the question respond with a message stating that question cannot be asnwered due to lack of information.\n",]
        question.append("Contexts:")
        question.append("\n".join([f'{i+1}. {context}' for i, context in enumerate(ranked_result)]))
        question.append(f'\nQuestion: {query}')
        answer = self.model.invoke("\n".join(question), top_p=0.9, temperature=0.1).content

        #answer = self.generator.generate(query, ranked_result, max_new_tokens=1024, top_p=0.9, temperature=0.1)
        return answer


class ImageDatabase:
    """
    A class to handle image data storage and retrieval using ChromaDB.
    """
    def __init__(self, collection):
        self.collection = collection
        logger.info("Initializing image database")
        logger.info("Loading CLIP model")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("Loading CLIP processor")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def add_data(self, images):
        metadata_list = []
        vector_list = []
        ids = []
        for i, image_path in enumerate(images, start = (len(self.collection.get()["ids"])+1)):
            image = Image.open(image_path)
            inputs = self.clip_processor(images=[image], return_tensors="pt")
            with torch.no_grad():
                vector_list.append(self.clip_model.get_image_features(**inputs).squeeze().tolist())
                metadata_list.append({"path": image_path})
                ids.append(str(i))

        self.collection.add(embeddings=vector_list, metadatas=metadata_list, ids=ids)

    def encode_clip_text(self,text):
        inputs = self.clip_processor(text=[text], return_tensors="pt")
        with torch.no_grad():
            vector = self.clip_model.get_text_features(**inputs).squeeze().tolist()
            return vector

    def answer_query(self,text_query):
        query_vector = self.encode_clip_text(text_query)
        search_result = self.collection.query(query_embeddings=[query_vector], n_results=1)
        return search_result if search_result else "No relevant image found"


class TableDatabase:
    """
    A class to handle table data storage and retrieval using ChromaDB.
    """
    def __init__(self, collection, model: BaseChatModel):
        logger.info("Initializing table database")
        self.collection = collection
        self.model = model
        logger.info("Loading Sentence Transformer model")
        self.text_model = get_sentence_transformer("mixedbread-ai/mxbai-embed-large-v1")
        logger.info("Loading MxbaiRetriever")
        self.retriever = MxbaiRetriever('mixedbread-ai/mxbai-embed-large-v1',collection)
        logger.info("Loading MxbaiRanker") 
        self.ranker = MxbaiReranker('mixedbread-ai/mxbai-rerank-large-v1')

    def encode_table_as_text(self, table):
        table_str = "\n".join([(" | ".join(map(str, row))).replace("\n","") for row in table])  # Convert to readable text
        return table_str

    def index(self, table, table_desc):
        table_str = self.encode_table_as_text(table)
        vector = self.text_model.encode(table_desc).tolist()
        metadata = {"table": table_str}
        self.collection.add(embeddings=[vector], metadatas=[metadata], ids=[str(len(self.collection.get()["ids"]) + 1)])

    def answer_query(self, query):
        query_vector = self.text_model.encode(query).tolist()
        result = self.collection.query(query_embeddings=[query_vector], n_results=1)
        retrieved_table = list(set([chunk_dict['table'] for chunk_dict in result['metadatas'][0]]))[0]
        table_str = [[row.split('|') for row in retrieved_table.split('\n')]]

        question = ["Use the table provided below and answer the question based on the table. The answer should be generated using the table only. If it seems insufficient to answer the question respond with a message stating that question cannot be asnwered due to lack of information."]
        question.append("Table:")
        question.append("\n".join([f'{i+1}. {context}' for i,context in enumerate(table_str)]))
        question.append(f'\nQuestion: {query}')
        answer = self.model.invoke("\n".join(question), top_p=0.9, temperature=0.1).content

        #answer = self.generator.generate(query, table_str, query_type='table', max_new_tokens=1024, top_p=0.9, temperature=0.1)
        return retrieved_table, answer 


def initialize_dbs(client: PersistentClient, model: BaseChatModel):
    """
    Initialize the databases with the specified collections.

    Returns the text, image, and table databases. 
    """
    text_ef = get_embedding_function(
        model_name="mixedbread-ai/mxbai-embed-large-v1"
    )

    # Define collections
    collections = {
        "text": client.get_or_create_collection("text_v1", embedding_function =text_ef),
        "images": client.get_or_create_collection("images_v1"),
        "tables": client.get_or_create_collection("tables_v1")
    }

    logger.info("Initializing text, image, and table databases")
    text_db = TextDatabase(collections["text"], model)
    logger.info("Text database initialized")

    logger.info("Initializing image database")
    image_db = ImageDatabase(collections["images"])
    logger.info("Image database initialized")

    logger.info("Initializing table database")
    table_db = TableDatabase(collections["tables"], model)
    logger.info("Table database initialized")

    return text_db, image_db, table_db

#
# Ingest pipeline code
#

def generate_table_description(table, openai_client: OpenAI):
    table_df = pd.DataFrame(table[1:], columns=table[0])
    markdown_table = table_df.to_markdown(index=False)
    prompt = f"""
    The following is a table extracted from a PDF:

    {markdown_table}

    Generate a 50 word summary about what are the contents of this table, as if writing a header to describe the table. Only output the description no extra text.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.6,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content



def create_database(text_db: TextDatabase, image_db: ImageDatabase, table_db: TableDatabase, document_folder='/content/documents'):
    """
    Create a database from the documents in the specified folder.
    """
    count=1
    for file_name in os.listdir(document_folder):
        if not file_name.endswith(".rst"):
            continue

        logger.info(f"- Ingesting file {count}: {file_name}")
        file_path = os.path.join(document_folder, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            rst_content = file.read()

        doctree = publish_doctree(rst_content)

        text_content = []
        tables = []
        images = []

        def is_inside_table(node):
            current_node = node
            while current_node.parent is not None:
                if hasattr(current_node.parent, 'tagname') and current_node.parent.tagname == 'table':
                    return True
                current_node = current_node.parent
                return False

        for node in doctree.traverse():
            if node.tagname == 'paragraph':
                if not is_inside_table(node):
                    text_content.append(node.astext())
                elif node.tagname == 'table':
                    rows = []
                    for row in node.traverse(condition=lambda x: x.tagname == 'row'):
                        cells = [cell.astext() for cell in row.traverse(condition=lambda x: x.tagname == 'entry')]
                        rows.append(cells)
                        tables.append(rows)
                elif node.tagname == 'image':
                    if 'uri' in node.attributes:
                        images.append(node.attributes['uri'][1:])


        output_dict = {}
        for i, text in tqdm(enumerate(text_content)):
            text_db.index(text)

        if len(images)>0:
            image_db.add_data(images)
            print(images)

        for i, table in tqdm(enumerate(tables)):
            table_description = generate_table_description(table)
            print(table_description)
            table_db.index(table, table_description)

        count+=1

    return text_db, image_db, table_db


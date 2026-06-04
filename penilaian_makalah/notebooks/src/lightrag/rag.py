import os
import asyncio
import logging

# from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger,logger ,wrap_embedding_func_with_attrs, set_verbose_debug

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"
    )
)

WORKING_DIR = os.path.join(
    BASE_DIR,
    "rag_storage_documents"
)

# env
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# Logger

setup_logger("lightrag", level="DEBUG")

# Config
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# LLM Model Function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        **kwargs,
    )

async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


# Lightrag
async def initialize_rag():

    embedding_dim = int(os.getenv("EMBEDDING_DIM", 1536))
    token_limit = int(os.getenv("EMBEDDING_TOKEN_LIMIT", 8192))
    model_name = os.getenv("EMBEDDING_MODEL")

    # Step 1: define raw embedding function
    async def raw_embedding_func(texts):
        return await openai_embed.func(
            texts,
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            model=model_name,
        )

    # Step 2: wrap embedding function
    embedding_func = wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=token_limit,
        model_name=model_name,
    )(raw_embedding_func)

    # Step 3: initialize LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage",
        enable_llm_cache=False, 
    )

    print("Initializing storages...")

    await rag.initialize_storages()
    return rag

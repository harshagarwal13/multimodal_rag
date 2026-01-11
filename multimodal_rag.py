import json
from typing import List

#Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

#Langchain imports
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()


def partition_document(file_path: str):
    """Exract elements from PDF using Unstructured"""
    elements = partition_pdf(
        filename=file_path,
        strategy="high_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    print(f"Extracted {len(elements)} elements from {file_path}")
    return elements


def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("Creating smart chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )
    print(f"Created {len(chunks)} chunks")
    return chunks

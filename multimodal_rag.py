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

def separate_content_types(chunk):
    """Analyze what type of chunk it has"""
    content_data = {
        "text": chunk.text,
        "images": [],
        "tables": [],
        "types": ['text'],
    }
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == "Table":
                content_data["types"].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                content_data["types"].append('image')
                if hasattr(element,'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data["types"].append('image')
                    content_data["images"].append(element.metadata.image_base64)
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text, tables, images):
    """Create an AI-enhanced summary of the text, tables, and images"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt_text= f"""You are creating a searchable description for content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT: {text}
        """
        if tables:
            prompt_text += f"TABLES:\n"
            for i , table in enumerate(tables):
                prompt_text += f"TABLE {i+1}:\n{table}\n\n"
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:
                1. Key facts, numbers and data points from texts and the tables
                2. Main topics and comcepts discussed
                3. Questions this content could answers
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms user might use.
                Make it detailed and searchable- prioritize searchability over brevity.

                SEARCHABLE DESCRIPTION:
                """
        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    
    except Exception as e:
        print(f"AI Summary failed: {e}")


def summarise_chunks(chunks):
    """Process all chunks with AI Summarise"""
    langchain_documents=[]
    total_chunks = len(chunks)
    for i,chunk in enumerate(chunks):
        current_chunk = i+1
        content_data = separate_content_types(chunk)
        if content_data['tables'] or content_data['images']:
            try:
                enhanced_content = create_ai_enhanced_summary(content_data["text"], content_data["tables"], content_data["images"])
            except Exception as e:
                print(f"Error in Enhancing summary: {e}")
                enhanced_content = content_data["text"]
        else:
            enhanced_content = content_data["text"]

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    "images_base64": content_data["images"]
                })
            }
        ) 

        langchain_documents.append(doc)
    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents
        
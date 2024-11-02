import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from config.globals import CHROMA_PATH
from get_embedding_function import get_embedding_function
from openai import OpenAI


PROMPT_TEMPLATE = """
Answer the question based only on the following context, 請用好笑的方式回答!!!!!:

{context}

---

Answer the question based on the above context: {question}
"""


model_used = "gpt-4o-mini"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

client = OpenAI(api_key=api_key)


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def get_openai_response(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = client.chat.completions.create(
            model = model_used,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in getting openai response: {e}")
        raise
        

def format_source(source_id: str) -> str:
    parts = source_id.split("\\")[-1].split(":")
    filename = parts[0]
    page = parts[1]
    return f"{filename} (page {page})"



def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(f"context_text: {context_text}")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"prompt: {prompt}")

    
    response_text = get_openai_response(prompt)

    sources = [format_source(doc.metadata.get("id", None)) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

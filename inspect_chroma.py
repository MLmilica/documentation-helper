"""Inspect documents stored in the local Chroma vector database."""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PERSIST_DIR = "chroma_db"
PREVIEW_CHARS = 200


def main() -> None:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    collection = vectorstore._collection
    data = collection.get(include=["documents", "metadatas"])

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    print(f"Database path: {PERSIST_DIR}/")
    print(f"Collection: {collection.name}")
    print(f"Total vectors: {len(ids)}")
    print("-" * 60)

    for i, doc_id in enumerate(ids, start=1):
        content = documents[i - 1] or ""
        metadata = metadatas[i - 1] or {}
        preview = content[:PREVIEW_CHARS].replace("\n", " ")
        if len(content) > PREVIEW_CHARS:
            preview += "..."

        print(f"\n[{i}] ID: {doc_id}")
        print(f"    Source: {metadata.get('source', 'N/A')}")
        print(f"    Length: {len(content)} chars")
        print(f"    Preview: {preview}")


if __name__ == "__main__":
    main()

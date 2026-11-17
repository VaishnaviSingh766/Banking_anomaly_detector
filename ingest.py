from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os


PDF_PATH = "knowledge_base/anomaly_rules_demo.pdf"
CHROMA_PATH = "./chroma_db"


def load_pdf():
    reader = PdfReader(PDF_PATH)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def create_chunks(text):
    # Simple chunking for prototype
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


def create_vector_database(chunks):

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")

    embeddings = model.encode(chunks)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="anomaly_rules"
    )

    # Avoid duplicate IDs
    ids = [
        f"rule_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": "anomaly_rules_demo.pdf"
            }
            for _ in chunks
        ]
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )


if __name__ == "__main__":

    if not os.path.exists(PDF_PATH):
        print(f"PDF not found: {PDF_PATH}")
        exit()

    print("Reading PDF...")

    text = load_pdf()

    print("Creating chunks...")

    chunks = create_chunks(text)

    print(f"Created {len(chunks)} chunks.")

    create_vector_database(chunks)

    print("RAG knowledge base created successfully!")
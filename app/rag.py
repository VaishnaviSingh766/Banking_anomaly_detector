import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "./chroma_db"


class RAGRetriever:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name="anomaly_rules"
        )

    def retrieve(self, query, top_k=3):

        query_embedding = self.model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        documents = results["documents"][0]

        return documents


if __name__ == "__main__":

    rag = RAGRetriever()

    results = rag.retrieve(
        "unusual transaction amount"
    )

    print("\nRAG RESULTS:\n")

    for result in results:
        print("--------------------")
        print(result)
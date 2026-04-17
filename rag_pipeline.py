import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PAPERS_DIR = 'papers'
CHROMA_DIR = 'chroma_db'
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_and_chunk_pdf():
    docs = []
    for filename in os.listdir(PAPERS_DIR):
        if filename.endswith('.pdf'):
            path = os.path.join(PAPERS_DIR, filename)
            loader = PyMuPDFLoader(path)
            pages = loader.load()
            print(f"Loaded: {filename} ---> {len(pages)} pages")
            docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks

def build_vectorstore():
    chunks = load_and_chunk_pdf()

    embeddings = HuggingFaceBgeEmbeddings(model_name = EMBED_MODEL)
    vectorstore =Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    print("Vectorstore saved to disk.")
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceBgeEmbeddings(model_name = EMBED_MODEL)
    vecorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vecorstore

def get_qa_chain(vectorstore):
    llm = ChatGroq(model=GROQ_MODEL, api_key=os.getenv('GROQ_API_KEY'), temperature=0.2)

    prompt = PromptTemplate.from_template("""
You are a research assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't have enough information from the provided papers."

Context:
{context}

Question: {question}

Answer:""")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser())
    return chain, retriever

def get_answer(query: str, chain, retriever) -> dict:
    answer = chain.invoke(query)
    source_docs = retriever.invoke(query)
    return {"answer": answer,
            "sources": [doc.metadata.get("source", "unknown") for doc in source_docs]}

if __name__ == "__main__":
    if not os.path.exists(CHROMA_DIR):
        vectorstore = build_vectorstore()
    else:
        print("Vectorstore already exists, loading from disk...")
        vectorstore = load_vectorstore()

    chain, retriever = get_qa_chain(vectorstore)

    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() == "exit":
            print("Bye!")
            break
        if not query:
            continue

        result = get_answer(query, chain, retriever)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {set(result['sources'])}")
        print("-" * 60)
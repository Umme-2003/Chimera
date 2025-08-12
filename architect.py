import os
import shutil # Used for safely removing directories
from dotenv import load_dotenv
from git import Repo # The GitPython library

# LangChain specific imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# NEW: A loader specifically for Python source code
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

# --- 1. Configuration and Setup ---
print("--- Phase 3: The Architect Agent ---")
load_dotenv()

# Configuration for the repository to be analyzed
REPO_URL = "https://github.com/gothinkster/flask-realworld-example-app.git"
REPO_PATH = "temp_repo"

# Initialize LLM and Embeddings model (reusing our setup from before)
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 2. Clone the Repository and Load Code ---
def load_and_vectorize_repo():
    """
    Clones a Git repository, loads all Python files, splits them into chunks,
    and creates a searchable vector store.
    """
    # Clean up any previous downloads
    if os.path.exists(REPO_PATH):
        print(f"Removing existing directory: {REPO_PATH}")
        shutil.rmtree(REPO_PATH)

    # Clone the repository from the URL
    print(f"Cloning repository from {REPO_URL}...")
    Repo.clone_from(REPO_URL, to_path=REPO_PATH)
    print("Repository cloned successfully.")

    # NEW: Use GenericLoader to load all Python files from the repo
    print("Loading Python files from the repository...")
    loader = GenericLoader.from_filesystem(
        path=REPO_PATH,
        glob="**/*.py", # Pattern to match all .py files
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=50),
        suffixes=[".py"]
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} Python documents.")

    # Split the loaded documents into smaller chunks for better RAG performance
    print("Splitting documents into chunks...")
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # Create and return the vector store
    print("Creating vector store from code chunks...")
    vector_store = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully.")
    return vector_store

# --- 3. Create the Q&A Chain ---
def create_qa_chain(vector_store):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain for answering questions
    about the codebase.
    """
    print("Creating Q&A chain...")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
    You are a senior software architect. You are an expert at analyzing large codebases.
    Answer the user's QUESTION based on the provided CONTEXT (snippets from the codebase).
    Provide a clear, concise answer, and reference the source file from the context if possible.

    CONTEXT:
    {context}

    QUESTION:
    {input}

    ANSWER:
    """)

    qa_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )
    return qa_chain

# --- 4. Main Execution ---
if __name__ == "__main__":
    # This is the main workflow
    db = load_and_vectorize_repo()
    qa_chain = create_qa_chain(db)

    # Example questions to ask the Architect Agent
    questions = [
        "What is the primary function of the 'app.py' file in the 'autoapp.py' module?",
        "How is user authentication handled in this project? Point to the relevant files.",
        "Are there any database models defined? If so, in which file?",
        "What is the purpose of the 'Article' model?",
    ]

    for question in questions:
        print(f"\n--- Asking Question ---")
        print(f"Question: {question}")

        # Invoke the chain to get the answer
        result = qa_chain.invoke({"input": question})

        print("\n--- Architect's Answer ---")
        print(result["answer"])
        print("------------------------\n")

    print("--- Architect analysis complete. ---")
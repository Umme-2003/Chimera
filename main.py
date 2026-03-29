import os
import re # NEW: Import the regular expressions library
from dotenv import load_dotenv

# --- 1. Setup ---
load_dotenv()

# LangChain specific imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LLM Selection: Gemini or Groq ---
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    print("Using LLM: Groq (Llama 3)")
elif os.getenv("GOOGLE_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    print("Using LLM: Google Gemini 2.0 Flash")
else:
    raise ValueError("No API key found! Set GROQ_API_KEY or GOOGLE_API_KEY in .env")

# --- Helper function to clean the LLM's code output ---
def extract_python_code(text: str) -> str:
    """
    Extracts Python code from a string, typically from an LLM response
    that includes Markdown formatting.
    """
    pattern = r"```(?:python\n)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loader = TextLoader("knowledge_base.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# --- 2. The Vulnerability Hunter Agent (Chain 1 - No changes) ---
print("Initializing Vulnerability Hunter Agent...")
prompt_hunter = ChatPromptTemplate.from_template("""
You are a cybersecurity analyst. Your task is to analyze the provided source code for vulnerabilities.
Use the following retrieved CONTEXT to explain the issue and propose a high-level fix.

CONTEXT:
{context}

SOURCE CODE:
```python
{input}
ANALYSIS & FIX PROPOSAL:
""")
document_chain_hunter = create_stuff_documents_chain(llm, prompt_hunter)
retrieval_chain_hunter = create_retrieval_chain(retriever, document_chain_hunter)
#--- 3. The Refactoring Engineer Agent (Chain 2 - No changes) ---
print("Initializing Refactoring Engineer Agent...")
prompt_engineer = ChatPromptTemplate.from_template("""
You are a senior software engineer specialized in secure coding.
Your goal is to modify the provided SOURCE CODE based on the provided ANALYSIS AND FIX PROPOSAL.
Your output must be the complete, corrected SOURCE CODE inside a Python markdown block.
Do not include any other explanation, notes, or analysis outside of the code block.
SOURCE CODE:
{input}
ANALYSIS AND FIX PROPOSAL:
{analysis_and_fix}
CORRECTED CODE:
""")
engineer_chain = prompt_engineer | llm | StrOutputParser()
#--- 4. The Orchestration (No changes to the logic) ---
print("Orchestrating agents...")
hunter_analysis = retrieval_chain_hunter | (lambda x: x['answer'])
engineer_input_formatter = RunnablePassthrough.assign(
analysis_and_fix=hunter_analysis
)
chimera_pipeline = engineer_input_formatter | engineer_chain
#--- 5. Run the Analysis and Generation ---
print("Loading the vulnerable code to be analyzed...")
with open("vulnerable_code.py", "r") as file:
    code_to_analyze = file.read()
print("Invoking the Chimera pipeline...")
raw_llm_output = chimera_pipeline.invoke({"input": code_to_analyze})
#--- NEW: Clean the output before printing and saving ---
print("Cleaning the generated code...")
corrected_code = extract_python_code(raw_llm_output)
print("\n--- AI AGENT GENERATED CORRECTED CODE (CLEANED) ---")
print(corrected_code)
#Save the cleaned code to a file
with open("vulnerable_code_corrected.py", "w") as f:
    f.write(corrected_code)
print("\n--- Clean code saved to vulnerable_code_corrected.py ---")
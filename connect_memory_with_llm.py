import os
import logging

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. Export your HuggingFace API token as HF_TOKEN before running this script."
    )

def load_llm(huggingface_repo_id):
    try:
        logger.info(f"Loading LLM from {huggingface_repo_id}")
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            max_new_tokens=512,
            huggingfacehub_api_token=HF_TOKEN,
        )
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        raise

# Step 2: Connect LLM with ChromaDB and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    try:
        logger.info("Setting custom prompt template")
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
        return prompt
    except Exception as e:
        logger.error(f"Error setting prompt: {str(e)}")
        raise

# Load Database
DB_CHROMA_PATH = "vectorstore/db_chroma"
try:
    logger.info(f"Loading embedding model and ChromaDB from {DB_CHROMA_PATH}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_model)
    logger.info("ChromaDB loaded successfully")
except Exception as e:
    logger.error(f"Error loading ChromaDB: {str(e)}")
    raise

# Create QA chain
try:
    logger.info("Creating RetrievalQA chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
except Exception as e:
    logger.error(f"Error creating QA chain: {str(e)}")
    raise

# Now invoke with a single query
try:
    user_query = input("Write Query Here: ")
    logger.info(f"Processing query: {user_query}")
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])
except Exception as e:
    logger.error(f"Error processing query: {str(e)}")
    raise
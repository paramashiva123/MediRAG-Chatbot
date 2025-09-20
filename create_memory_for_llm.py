import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"
def load_pdf_files(data):
    try:
        logger.info(f"Loading PDFs from {data}")
        loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} PDF pages")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDFs: {str(e)}")
        raise

documents = load_pdf_files(data=DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data):
    try:
        logger.info("Creating text chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(extracted_data)
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        raise

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings 
def get_embedding_model():
    try:
        logger.info("Loading embedding model")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embedding_model
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise

embedding_model = get_embedding_model()

# Step 4: Store embeddings in ChromaDB
DB_CHROMA_PATH = "vectorstore/db_chroma"
try:
    logger.info(f"Creating ChromaDB vector store at {DB_CHROMA_PATH}")
    db = Chroma.from_documents(text_chunks, embedding_model, persist_directory=DB_CHROMA_PATH)
    db.persist()
    logger.info("Vector store created and persisted successfully")
except Exception as e:
    logger.error(f"Error creating or persisting vector store: {str(e)}")
    raise
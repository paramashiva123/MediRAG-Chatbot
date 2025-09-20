import os
import streamlit as st
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CHROMA_PATH = "vectorstore/db_chroma"

@st.cache_resource
def get_vectorstore():
    try:
        logger.info("Loading embedding model and ChromaDB")
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_model)
        logger.info("Vector store loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

def set_custom_prompt(custom_prompt_template):
    try:
        logger.info("Setting custom prompt template")
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
        return prompt
    except Exception as e:
        logger.error(f"Error setting prompt: {str(e)}")
        raise

# Custom CSS for professional styling
st.markdown(
    """
    <style>
    .main {
        background-color: #E6F3E6; /* Light green background */
        /* Uncomment the line below and comment the above for light yellow: */
        /* background-color: #FFF9E6; */
        padding: 20px;
    }
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .title-image {
        margin-right: 20px;
        max-width: 200px;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Sidebar for additional options (optional, can be expanded)
    st.sidebar.title("MediBot Settings")
    st.sidebar.write("Configure your chatbot experience here.")

    # Title and image
    st.markdown(
        '<div class="title-container">'
        '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==" class="title-image" alt="MediBot Header">'
        '<h1 style="color: #1E90FF;">MediBot: AI Medical Chatbot using RAG</h1>'
        '</div>',
        unsafe_allow_html=True
    )
    # Note: Replace the base64 placeholder with the actual image data if needed.
    # For now, assume the image is saved as 'header_image.png' in the project directory.
    # Use st.image() instead if the image is a file:
    st.image("medibot.png", width=200, caption="MediBot Interface")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat container for a professional look
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        prompt = st.chat_input("Ask a medical question...")

        if prompt:
            with st.chat_message('user'):
                st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            CUSTOM_PROMPT_TEMPLATE = """
                    Use the pieces of information provided in the context to answer user's question.
                    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                    Dont provide anything out of the given context

                    Context: {context}
                    Question: {question}

                    Start the answer directly. No small talk please.
                    """
            
            try: 
                logger.info("Retrieving vector store")
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                logger.info("Creating RetrievalQA chain")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                        temperature=0.0,
                        groq_api_key=os.environ["GROQ_API_KEY"],
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                logger.info(f"Processing query: {prompt}")
                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                source_documents = response["source_documents"]
                result_to_show = result + "\n**Source Docs:**\n" + str(source_documents)
                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                logger.error(f"Error in main processing: {str(e)}")
                st.error(f"Error: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# MediBot: AI Medical Chatbot using RAG

## Overview
MediBot is an advanced AI-powered medical chatbot designed to provide context-aware responses using Retrieval-Augmented Generation (RAG). Built with Python, it leverages a combination of modern AI frameworks, large language models (LLMs), and vector databases to process medical documents (e.g., PDFs) and answer user queries accurately. This project is ideal for educational purposes or as a proof-of-concept for medical knowledge retrieval, though it is not intended for clinical decision-making.

The chatbot integrates with LangChain, HuggingFace, and Groq-hosted LLMs, utilizing ChromaDB as the vector store for efficient document embedding and retrieval. The user interface is developed using Streamlit, offering a professional and interactive experience.

## Project Layout
The development of MediBot is structured into three phases:

### Phase 1: Setup Memory for LLM (Vector Database)
- **Load raw PDF(s)**: Import medical documents from the `data/` directory.
- **Create Chunks**: Split documents into manageable text chunks using LangChain's `RecursiveCharacterTextSplitter`.
- **Create Vector Embeddings**: Generate embeddings using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model.
- **Store Embeddings in ChromaDB**: Persist embeddings in a ChromaDB vector store for efficient retrieval.

### Phase 2: Connect Memory with LLM
- **Setup LLM (Mistral with HuggingFace)**: Initialize a Mistral-7B-Instruct-v0.3 model via HuggingFace Endpoint or a Groq-hosted model (e.g., `meta-llama/llama-4-maverick-17b-128e-instruct`).
- **Create Chain with ChromaDB**: Build a RetrievalQA chain to connect the LLM with the ChromaDB vector store, enabling context-aware responses.

### Phase 3: Setup UI for the Chatbot
- **Chatbot with Streamlit**: Develop an interactive UI with a light green background, professional layout, and a header image.
- **Load Vector Store (ChromaDB) in Cache**: Utilize Streamlit's caching to load the vector store efficiently.
- **Retrieval Augmented Generation (RAG)**: Implement RAG to retrieve relevant document chunks and generate responses.

## Tools & Technologies
- **LangChain**: AI framework for LLM applications.
- **HuggingFace (ML/AI Hub)**: Provides pre-trained models and embeddings.
- **Mistral (LLM Model)**: Used as the primary language model.
- **ChromaDB (Vector Database)**: Replaces FAISS for storing and retrieving vector embeddings.
- **Streamlit**: Framework for the chatbot UI.
- **Python**: Programming language.
- **VS Code**: Integrated Development Environment (IDE).

## Technical Architecture
The architecture is divided into three phases:
- **Phase 1**: Knowledge source (PDFs) is processed into chunks, embedded using HuggingFace, and stored in ChromaDB.
- **Phase 2**: The LLM (via LangChain) performs semantic search on ChromaDB to retrieve relevant embeddings, feeding them into the generation pipeline.
- **Phase 3**: Streamlit UI accepts user queries, passes them to the RAG pipeline, and displays the LLM's response with source documents.

## Improvement Potential/Next Steps
- **Add Authentication in the UI**: Implement user login for secure access.
- **Make Use of Self-Upload Document Functionality**: Allow users to upload PDFs and embed them dynamically.
- **Add Multiple Documents and Embed Them Together**: Support processing multiple documents in a single session.
- **Add Unit Testing of RAG Applications**: Develop tests to ensure RAG pipeline reliability.

## Summary
- **Modern AI Chatbot for Documents**: MediBot leverages cutting-edge RAG technology.
- **Modular 3-Phased Chatbot Project**: Structured into memory setup, LLM integration, and UI development.
- **Talked About**:
  - Streamlit
  - LangChain | HuggingFace
  - RAG
  - Vector Embeddings
  - End-to-End RAG Pipeline
- **Future Possibilities**: Everything (code and all dependencies) will be shared in the comments section.
- **Any Sort of Feedback will be Highly Appreciated**: Contributions and suggestions are welcome.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/MediRAG-Chatbot.git
   cd MediRAG-Chatbot
   ```
2. **Set Up Virtual Environment**:
   ```bash
   uv venv
   .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
   - Ensure Microsoft Visual C++ Build Tools are installed for `chroma-hnswlib` (https://visualstudio.microsoft.com/visual-cpp-build-tools/).
4. **Configure Environment Variables**:
   - Create a `.env` file with your HuggingFace API token (`HF_TOKEN`) and Groq API key (`GROQ_API_KEY`).
   - Example:
     ```
     HF_TOKEN=your_hf_token
     GROQ_API_KEY=your_groq_api_key
     ```
5. **Prepare Data**:
   - Place PDF files in the `data/` directory.

## Usage
1. **Build Vector Store**:
   ```bash
   python create_memory_for_llm.py
   ```
2. **Run CLI Demo**:
   ```bash
   python connect_memory_with_llm.py
   ```
   - Enter a query (e.g., "What is diabetes?") to test.
3. **Launch Web UI**:
   ```bash
   streamlit run medibot.py
   ```
   - Access the app at `http://localhost:8501`.

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. For major changes, open an issue first to discuss.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Inspired by the need for accessible medical knowledge retrieval.
- Built with open-source tools from HuggingFace, LangChain, and Streamlit communities.
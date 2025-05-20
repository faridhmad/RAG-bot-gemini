# RAG Bot with Google Gemini & LangChain

This project implements a Retrieval Augmented Generation (RAG) system using Google's Gemini Pro model (via `gemini-2.0-flash`) and the LangChain framework. It allows you to chat with your own text documents by:

1.  Loading text files from a specified folder.
2.  Splitting the documents into manageable chunks.
3.  Generating embeddings for these chunks using HuggingFace's `all-MiniLM-L6-v2` model.
4.  Storing these embeddings in a FAISS vector store for efficient similarity search.
5.  Using LangChain's `RetrievalQA` chain to retrieve relevant document chunks based on a user's query and then generate an answer using the Gemini model.

The system provides a simple command-line interface for interaction, simulating a customer service bot.

[![GitHub stars](https://img.shields.io/github/stars/faridhmad/RAG-bot-gemini.svg?style=social&label=Star&maxAge=2592000)](https://github.com/faridhmad/RAG-bot-gemini/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/faridhmad/RAG-bot-gemini.svg?style=social&label=Fork&maxAge=2592000)](https://github.com/faridhmad/RAG-bot-gemini/network/members)

## Features

*   **Google Gemini Integration**: Leverages the powerful `gemini-2.0-flash` model for generation.
*   **Local Document Processing**: Ingests `.txt` files from a local directory.
*   **Efficient Embeddings**: Uses `all-MiniLM-L6-v2` for creating compact and effective text embeddings.
*   **FAISS Vector Store**: In-memory vector store for fast retrieval of relevant document chunks.
*   **LangChain Powered**: Built using various LangChain components for a modular RAG pipeline.
*   **Environment Variable Management**: Securely loads API keys using `python-dotenv`.
*   **Interactive CLI**: Simple command-line interface for asking questions.
*   **M1 Mac Compatibility**: Includes specific versions of `protobuf` and `grpcio` for better compatibility on Apple Silicon Macs.

## How it Works (RAG Pipeline)

1.  **Load Documents**: The script scans a specified folder (`document_folder`) for `.txt` files and loads their content.
2.  **Split Text**: The loaded documents are split into smaller chunks using `CharacterTextSplitter` to ensure they fit within the model's context window and improve retrieval relevance.
3.  **Generate Embeddings**: Each text chunk is converted into a numerical vector (embedding) using the `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.
4.  **Create Vector Store**: The embeddings and their corresponding text chunks are stored in a FAISS vector store. FAISS allows for efficient similarity searches.
5.  **User Query**: The user inputs a question via the command-line interface.
6.  **Retrieve Relevant Chunks**: The user's query is also embedded, and FAISS is used to find the most semantically similar document chunks from the vector store.
7.  **Augment Prompt & Generate Answer**: The retrieved chunks (context) are combined with the user's original query to form a prompt for the Gemini LLM. The LLM then generates an answer based on the provided context and its general knowledge.
8.  **Display Answer**: The generated answer is displayed to the user.

## Prerequisites

*   Python 3.9+
*   A Google Cloud Project with the Generative Language API (Gemini API) enabled.
*   A `GOOGLE_API_KEY` for accessing the Gemini API.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/faridhmad/RAG-bot-gemini.git
    cd RAG-bot-gemini
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project:
    ```
    .env
    ```
    Add your Google API key to the `.env` file:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```
    Replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual API key.

5.  **Prepare your documents:**
    *   The script looks for documents in a folder specified by the `document_folder` variable in `rag_gemini_m1.py`.
    *   By default, this is set to `document_folder = "/your-path-to-folder-documents"`.
    *   **You MUST change this path** to a folder on your system where your `.txt` documents are located.
        For example, if you create a folder named `my_data` in the project root:
        ```
        RAG-bot-gemini/
        ├── my_data/
        │   ├── doc1.txt
        │   └── doc2.txt
        ├── rag_gemini_m1.py
        ...
        ```
        Then, update line 15 in `rag_gemini_m1.py` to:
        ```python
        document_folder = "my_data"
        ```
    *   Place your `.txt` files inside this folder.

## Usage

Once the setup is complete, run the script:

```bash
python rag_gemini_m1.py
```


You will be greeted by the "ABC Customer Service System" prompt. Type your questions and press Enter. To exit, type exit or quit.

```bash
=== Welcome to ABC Customer Service System ===
We are here to assist you with any inquiries!
Type 'exit' to end the session.


How can we assist you today? What are the refund policies?

Please hold on, we are searching for an answer for you...

🤖 Our answer: [Answer generated by Gemini based on your documents]

Is there anything else we can assist you with?
```

```
Directory Structure
RAG-bot-gemini/
├── .env                         # For API keys (you need to create this)
├── your-path-to-folder-documents/ # Example: my_data/ (contains your .txt files)
│   └── example_document.txt
├── rag_gemini_m1.py             # Main application script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

M1 Mac Compatibility Notes

The requirements.txt file includes specific versions for protobuf and grpcio:
```
protobuf==4.25.3
grpcio==1.62.1
```

These versions are known to work well on Mac M1/M2/M3 systems and can help resolve potential installation or runtime issues related to gRPC, which is a dependency for Google's client libraries.

Customization & Future Enhancements

Support for more document types: Extend TextLoader with other loaders like PyPDFLoader, CSVLoader, etc.

Different Embedding Models: Experiment with other sentence transformer models or OpenAI embeddings.

Different LLMs: Swap ChatGoogleGenerativeAI with other LangChain-compatible LLMs (e.g., from OpenAI, HuggingFace Hub).

Persistent Vector Store: Save the FAISS index to disk (db.save_local("faiss_index")) and load it (FAISS.load_local("faiss_index", embedding)) to avoid re-processing documents on every run.

More Sophisticated Chain Types: Explore other chain types like map_reduce, refine, or map_rerank for handling larger documents or more complex queries.

Web Interface: Build a web UI using Streamlit or Flask.

Error Handling: Add more robust error handling, especially for API calls and file operations.

Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, features, or improvements.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

License

This project is licensed under the MIT License. See the LICENSE file for details (if you choose to add one).

This README provides a good starting point. You might want to add a LICENSE file (e.g., MIT License is common for such projects).

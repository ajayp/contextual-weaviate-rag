 # Local RAG System with Weaviate, Ollama, and LlamaParse

 This project demonstrates a complete local Retrieval-Augmented Generation (RAG) system. It leverages:
 - **Weaviate (v4)** as the vector database.
 - **Ollama** for local embedding (`nomic-embed-text`) and generative model (`llama3.2`).  
 - **LlamaParse** for intelligent document parsing.

 This proof of concept is designed to process documents, chunk them while preserving page and structural information, store them in Weaviate, and then answer questions using the retrieved context.

<img width="1207" alt="image" src="https://github.com/user-attachments/assets/f619be37-3802-4042-90eb-03ad2d39a544" />

 ## Prerequisites

 Before you begin, ensure you have the following installed and running:

 ### 1. Docker
 [Docker](https://docker.com) or [OrbStack](https://orbstack.dev) is required to run the Weaviate vector database.
 - OrbStack is faster and more efficient. 
 - Ensure it is running before starting the Weaviate container.

 ### 2. Ollama
 Ollama is used to run large language models locally for both embedding (vectorization) and generation.
 - Download and install [Ollama](https://ollama.com/download) for your machine.  
 - Once installed, ensure Ollama is running.

 ### 3. LlamaParse API Key
 LlamaParse is used for intelligent parsing of PDF documents, extracting text, tables, and other structured data while maintaining page integrity.
 - Obtain an API key from the Llama Cloud website.
 - Set your LlamaParse API key as an environment variable named `LLAMAPARSE_API_KEY`:
   ```bash
   export LLAMAPARSE_API_KEY="your_llamaparse_api_key_here"
   ```

 ## Setup

 1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

 2. **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

 3. **Install Python dependencies:**
   
    ```bash
    pip install -r requirements.txt
    ```

 4. **Start Weaviate using Docker Compose:**

    ```bash
    docker compose up -d
    ```
    Wait a moment for the Weaviate container to fully start.

 ## Running the Application

 1. **Ensure Ollama is running:**
    The `main.py` script will attempt to pull the required models (`nomic-embed-text` and `llama3.2`) if they are not already downloaded.

 2. **Run the main script:**
    ```bash
    python main.py
    ```

    The script will:
    - Connect to Ollama and Weaviate.
    - Create the `DocumentChunk` collection in Weaviate (or use an existing one).
    - If no documents are found in Weaviate, it will process the example PDF (`reinsurance-agreement.pdf`) using LlamaParse, chunk it, and import it into Weaviate.
    - Start an interactive Q&A session.

 ## Usage

 Once the application is running, you can interact with it via the command line:

 - **Regular Question:**
   ```
   💬 Ask a question (or 'quit' to exit): What are the coverage limits?
   ```
   
 To exit the interactive session, type `quit`, `exit`, or `q`.

 ## Cleaning Up

 To stop and remove the Weaviate container:
 ```bash
 docker compose down
 ```
 To remove the Ollama models (if desired):
 ```bash
 ollama rm nomic-embed-text
 ollama rm llama3.2
 ```

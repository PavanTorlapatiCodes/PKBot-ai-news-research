# PKBot – AI-Powered News Research Tool

PkBot is an AI-powered news research application that allows users to process news article URLs and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant article information and generate answers with source references.

## Features

- Process multiple news article URLs.
- Extract article text and split it into smaller chunks.
- Generate semantic embeddings using Hugging Face Sentence Transformers.
- Store and retrieve document embeddings using FAISS.
- Generate answers using a Groq-hosted language model.
- Display answers and source references through an interactive Streamlit interface.

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM Provider:** Groq API
- **LLM Integration:** LangChain and `langchain-groq`
- **Document Loading:** UnstructuredURLLoader
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector Database:** FAISS
- **Architecture:** Retrieval-Augmented Generation (RAG)

## How It Works

1. The user enters one or more news article URLs.
2. The application extracts text from the web pages.
3. The text is split into smaller, overlapping chunks.
4. Hugging Face generates embeddings for the chunks.
5. FAISS stores the embeddings for efficient similarity search.
6. The user submits a question about the articles.
7. FAISS retrieves relevant chunks, and LangChain passes the retrieved context to the Groq language model.
8. The application displays the generated answer and available source references.

## Project Structure

```text
PKbot-ai-news-research/
├── newsres.py
├── README.md
├── requirements.txt
```

The FAISS index directory is generated when the application processes articles.

## Prerequisites

- Python 3.10 or newer
- A Groq API key
- Internet access for the Groq API and initial embedding-model download

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rockybot-ai-news-research.git
cd rockybot-ai-news-research
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure your Groq API key as an environment variable:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Update `newsresearch.py` to read the key securely using `os.getenv("GROQ_API_KEY")` instead of hardcoding it.

## Run the Application

```bash
python -m streamlit run newsresearch.py
```

Open the local URL shown in the terminal, enter news article URLs, process them, and submit a question.

## Security

- Never commit API keys or other secrets.
- Add `.env`, `.venv/`, Python cache files, and generated vector indexes to `.gitignore` as appropriate.
- Only load FAISS indexes created by a trusted source.

## Limitations

- Some websites block automated extraction or require JavaScript rendering.
- Answer quality depends on the extracted article content and retrieved context.
- Groq model availability and API limits depend on the account and the selected model.
- The local embedding model requires an initial download.

## Learning Outcomes

This project demonstrates practical implementation of LLM APIs, RAG pipelines, semantic search, embeddings, vector storage, document processing, and interactive AI applications.

## Future Improvements

- Support PDF and additional document formats.
- Add chat history and persistent user sessions.
- Improve URL extraction and error handling.
- Deploy the application to a cloud platform.
- Add automated tests and configuration management.

## Author
T Pavan Kumar
Developed as a hands-on project in Generative AI and AI application development.

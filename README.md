# LangChain Examples

This folder contains example scripts demonstrating the use of LangChain agents, tools, and integrations for building AI-powered assistants and retrieval-augmented generation (RAG) systems.

## Contents

- `tuto.py` — Weather assistant using Google Gemini and custom tools for weather and location lookup.
- `rag.py` — Retrieval-augmented generation (RAG) agent with a small knowledge base about computers and fruits.
- `multimodalInput.py` — (Not described here; add details if needed.)
- `.env` — Environment variables (API keys, etc.).
- `.gitignore` — Ignore rules for Python, environments, and editor files.

## Setup

1. **Install dependencies** (from the project root):
   ```bash
   pip install -r requirements.txt
   # or, if using pyproject.toml
   pip install .
   ```
2. **Configure environment variables**:
   - Copy `.env.example` to `.env` and fill in your API keys (Google Gemini, Groq, etc.).

## Usage

### Weather Assistant

Run the weather assistant example:

```bash
python tuto.py
```

- Answers weather questions, fetches real-time weather data, and locates user cities.

### RAG Example

Run the RAG agent example:

```bash
python rag.py
```

- Answers questions about computers, laptops, Apple products, and fruits using a small in-memory knowledge base.

### Dynamic Prompt example

Run the dynamic prompt agent example

```bash
python dynamic_prompt.py
```

## Requirements

- Python 3.9+
- [LangChain](https://python.langchain.com/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [requests](https://pypi.org/project/requests/)
- [sentence-transformers](https://www.sbert.net/)
- [faiss-cpu](https://pypi.org/project/faiss-cpu/)
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/)
- [langchain-groq](https://pypi.org/project/langchain-groq/)
- [langchain-huggingface](https://pypi.org/project/langchain-huggingface/)

## Notes
- I have used uv for creating environment and installing packages
- Place your API keys in the `.env` file.
- The weather tool uses the public [wttr.in](https://wttr.in/) API.

## License

MIT License

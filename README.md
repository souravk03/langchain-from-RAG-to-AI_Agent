# langchain-from-RAG-to-AI_Agent

# LangChain Practice

A hands-on collection of LangChain fundamentals, built while learning the framework end-to-end — from core building blocks to a working RAG chatbot.

## What's covered

- **Models** — LLMs, chat models (OpenAI, Google, Anthropic, HuggingFace API & local), and embedding models with document similarity
- **Prompts** — prompt templates, chat prompt templates, message placeholders, and a simple prompt-testing UI
- **Chains** — simple, sequential, parallel, and conditional chains
- **Runnables** — RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
- **Output Parsers** — string, JSON, structured, and Pydantic output parsers
- **Structured Output** — typed dict, Pydantic, and JSON-schema based structured outputs
- **Document Loaders** — text, CSV, PDF, directory, and web-based loaders
- **Text Splitters** — length-based, structure-based, markdown-based, Python-code-based, and semantic-meaning-based splitting
- **Retrievers & Vector Stores** — FAISS-based vector store and retriever examples
- **Tools & Tool Calling** — custom tool creation and tool-calling workflows
- **Agents** — basic agent construction and execution
- **Project: YouTube RAG Chatbot** — a retrieval-augmented chatbot that answers questions about a YouTube video's transcript, with a pre-built FAISS index

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root with your API keys (OpenAI, Anthropic, Google, HuggingFace as needed):

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_key_here
```

## Notes

This is a learning/reference repo — each file is a self-contained example of one LangChain concept rather than a single production application, aside from the YouTube RAG chatbot project.

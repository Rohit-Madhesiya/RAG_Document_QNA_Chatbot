# Conversational RAG with PDF Search

## Overview 📄🔍🤖
This is a Streamlit-based application that allows users to upload PDF documents and interact with them using a Conversational RAG (Retrieval-Augmented Generation) approach. It utilizes LangChain, FAISS, and Chroma for document retrieval and embedding, along with the Groq API for LLM-powered responses.

## Features ✨📚💡
- **Conversational RAG:** Ask questions about uploaded PDFs and get accurate responses.
- **PDF Upload & Parsing:** Users can upload PDFs, which are automatically processed.
- **Efficient Document Search:** Uses FAISS and Chroma for fast retrieval.
- **Chat History Management:** Maintains session-based conversation history.
- **Groq API Integration:** Leverages the `deepseek-r1-distill-llama-70b` model for answering queries.

## Installation 🛠️💻📦

1. Clone the repository:
   ```bash
   git clone https://github.com/Rohit-Madhesiya/RAG_Document_QNA_Chatbot.git
   cd RAG_Document_QNA_Chatbot
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add the following lines:
     ```env
     GROQ_API_KEY=<your_api_key>
     OPENAI_API_KEY=<your_openai_api_key>
     HUGGING_FACE_ACCESS_TOKEN=<your_huggingface_token>
     ```

## Usage 🚀🗂️💬

Run the Streamlit app with the following command:
```bash
streamlit run app.py
```

Once the app is running:
- Enter your Groq API key.
- Upload a PDF document.
- Ask questions about the document and receive relevant answers.
- View document similarity searches in the expander section.

## Dependencies 🏗️📌📝
The project requires the following libraries (listed in `requirements.txt`):
- `streamlit`
- `langchain`
- `langchain_core`
- `langchain_community`
- `langchain_groq`
- `langchain_ollama`
- `langchain_huggingface`
- `langchain_chroma`
- `python-dotenv`

## Project Structure 📁🛠️🔍
```
├── app.py              # Main application file for RAG Q&A
├── main.py             # Conversational RAG with chat history
├── requirements.txt    # List of dependencies
├── .env                # API key configuration (not included in repo)
└── temp_pdf/           # Temporary directory for uploaded PDFs
```

## Contributing 🤝🔧📢
Feel free to fork the repository and submit pull requests for improvements!

## Author ✍️👨‍💻🚀
Developed by [Rohit Gupta].


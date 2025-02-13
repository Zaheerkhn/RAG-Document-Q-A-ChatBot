# RAG Document Q&A ChatBot

![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-Apache_2.0-orange)

## 📌 Overview
This project is a **RAG (Retrieval-Augmented Generation) Document Q&A ChatBot**, built using **Streamlit, LangChain, FAISS, and HuggingFace Embeddings**. It allows users to **upload a PDF document, generate embeddings**, and ask questions based on the document's content.

## 🚀 Features
- 📂 **Upload PDF Documents**
- 🔍 **Generate Vector Embeddings** using **FAISS**
- 🏆 **Retrieve Relevant Information** from documents
- 💡 **Answer Queries Based on Uploaded PDF**
- 🤖 **Uses Llama-3.3-70b-Specdec Model via Groq API**

## 🛠️ Installation
### **Clone the Repository**
```bash
 git clone https://github.com/Zaheerkhn/RAG-Document-Q-A-ChatBot.git
 cd RAG-Document-Q-A-ChatBot
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Set Up Environment Variables**
Create a `.env` file in the project root and add your API keys:
```
HF_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langchain_api_key
GROQ_API_KEY=your_groq_api_key
```

## 🎯 Usage
Run the Streamlit app with the command:
```bash
streamlit run app.py
```

## 📜 License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Feel free to submit **issues, feature requests, or pull requests** to improve this project!

## 📞 Contact
For any queries, reach out via GitHub Issues.

---
🚀 **Developed by [Zaheerkhn](https://github.com/Zaheerkhn)**


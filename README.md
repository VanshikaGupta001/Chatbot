<h1>RoBoDoc: AI-Powered Medical Assistant</h1>

<i>Name: Vanshika Gupta
Roll number: cse240001076</i>

RoBoDoc is an AI-driven chatbot designed to simulate real doctor interactions. Built with Streamlit, Groq API, FAISS, and speech processing libraries, it predicts diseases based on user symptoms and provides medical guidance.

<b>Features</b>

Speech-to-Text & Text-to-Speech: Converts user voice input to text and AI responses to speech.

Retrieval-Augmented Generation (RAG): Uses FAISS to retrieve relevant medical knowledge.

Conversational Memory: Maintains context for interactive diagnosis.

API Integration: Utilizes Groq’s LLaMA-3 model for intelligent responses.

<b>Workflow</b>

User provides symptoms via text or speech.

AI retrieves relevant knowledge and generates a diagnosis.

Response is displayed as text and spoken aloud.

Conversation history is maintained for contextual engagement.

<b>How to Run</b>

<u>Clone the repository:</u>

git clone https://github.com/VanshikaGupta001/Chatbot

<u>Install dependencies:</u>

pip install -r requirements.txt

<u>Run the Streamlit app:</u>

streamlit run robodoc.py

<b>Conclusion</b>

RoBoDoc enhances AI-driven medical interactions using RAG, speech processing, and chatbot intelligence. Future improvements include refining FAISS indexing with real medical data for greater accuracy.

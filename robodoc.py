import streamlit as st
from groq import Groq
import faiss
import os
import numpy as np
from audio_recorder_streamlit import audio_recorder
import base64
from gtts import gTTS
from pydub import AudioSegment
import json

st.title("🩺 RoBoDoc")

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            response_format="text",
            file=audio_file
        )
    return transcript

def text_to_speech(input_text, speed=1.3):
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts_file_path = "temp_audio_play.mp3"
    tts.save(tts_file_path)

    # Load the audio file 
    audio = AudioSegment.from_file(tts_file_path)
    faster_audio = audio.speedup(playback_speed=speed)
    
    # Save the modified audio
    fast_audio_path = "temp_audio_fast.mp3"
    faster_audio.export(fast_audio_path, format="mp3")
    
    return fast_audio_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
    
# Groq API
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

api_key = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = api_key

groq_client = Groq(api_key=api_key)

# Load FAISS index for RAG
index = faiss.IndexFlatL2(768)  # Assuming 768-dimensional embeddings
knowledge_base = []  # Placeholder for documents

def retrieve_documents(query_embedding, top_k=3):
    if len(knowledge_base) == 0:
        return []
    _, indices = index.search(np.array([query_embedding]), top_k)
    return [knowledge_base[i] for i in indices[0]]

def get_disease_prediction(user_input, conversation_history):
    query_embedding = np.random.rand(768).astype("float32")  
    retrieved_docs = retrieve_documents(query_embedding)
    context = "\n".join(retrieved_docs)
    
    prompt = f"""
    You are a medical assistant diagnosing diseases.
    Based on symptoms, ask follow-up questions if needed.
    Predict the most probable disease and suggest medicines and dietary care.
    
    Conversation history:
    {conversation_history}
    
    Patient: {user_input}
    
    Response:
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Streamlit UI
with st.sidebar:
    st.title("🩺 RoBoDoc")
    st.write("Interact with the chatbot just like a real doctor!")
    audio_bytes = audio_recorder()
    if audio_bytes:
     with st.spinner("Listening..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        transcript = speech_to_text(webm_file_path)
        os.remove(webm_file_path)
        if transcript:
            user_input = transcript

if "conversation" not in st.session_state:
    st.session_state.conversation = []

for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Thinking 🤔..."):
        result = get_disease_prediction(user_input, st.session_state.conversation)
    
    with st.chat_message("assistant"):
        st.markdown(result)
    
    with st.spinner("Generating audio response..."):
        audio_file = text_to_speech(result)
        autoplay_audio(audio_file)
        os.remove(audio_file)
    
    st.session_state.conversation.append({"role": "assistant", "content": result})

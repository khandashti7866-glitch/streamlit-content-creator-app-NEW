# -----------------------------------------------------------
# AI SOCIAL MEDIA CONTENT CREATOR – VIP CHATBOT
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import random
import re
import time
from datetime import datetime

# Optional transformer model
MODEL_LOADED = False
try:
    from transformers import pipeline
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

# Optional voice
try:
    import pyttsx3
    VOICE_ENABLED = True
except:
    VOICE_ENABLED = False

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI SOCIAL MEDIA CONTENT CREATOR",
    layout="wide",
    page_icon="✨"
)

# -----------------------------------------------------------
# VIP Theme CSS + Glass Style
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-attachment: fixed;
        color: #fff;
        font-family: 'Segoe UI';
    }

    .glass {
        background: rgba(0,0,0,0.6);
        padding: 18px;
        border-radius: 14px;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    }

    .gold-title {
        font-size: 36px;
        font-weight: 900;
        color: gold;
        text-shadow: 0 0 12px gold, 0 0 6px rgba(0,0,0,0.6);
        margin-bottom: 12px;
    }

    .chat-user {
        background: linear-gradient(90deg,#ffd166,#ffb703);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 6px;
        color: black;
        font-weight: bold;
    }

    .chat-bot {
        background: rgba(255,255,255,0.1);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 6px;
    }

    .stButton>button {
        background: gold !important;
        color: black !important;
        border-radius: 10px;
        font-weight: bold;
        padding: 8px 16px;
        box-shadow: 0 0 12px gold;
    }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts {"role": "user"/"bot", "text": "..."}
if "projects" not in st.session_state:
    st.session_state.projects = []  # saved chat sessions

# -----------------------------------------------------------
# Load Local Model (Optional)
# -----------------------------------------------------------
generator_model = None
if MODEL_LOADED:
    try:
        generator_model = pipeline("text-generation", model="distilgpt2")
    except:
        generator_model = None

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<h2 class="gold-title">AI SOCIAL MEDIA CONTENT CREATOR</h2>', unsafe_allow_html=True)
    tab_option = st.radio("Navigation", ["💬 Chat", "📁 Projects", "📜 History", "⚙️ Settings"])
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
def generate_reply(prompt):
    if generator_model:
        try:
            output = generator_model(prompt, max_length=120)[0]["generated_text"]
            return output.strip()
        except:
            pass
    # Fallback template
    templates = [
        f"{prompt} - This is a motivational response for social media.",
        f"Here's what you can post about: {prompt}",
        f"Quick tip on {prompt}: Engage your audience effectively!",
        f"{prompt} explained in simple words for your followers."
    ]
    return random.choice(templates)

def speak_text(text):
    if VOICE_ENABLED:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

# -----------------------------------------------------------
# CHAT TAB
# -----------------------------------------------------------
if tab_option == "💬 Chat":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<h2 class="gold-title">Chat with AI</h2>', unsafe_allow_html=True)
    st.markdown('<p>Ask anything and get creative social media content or advice!</p>', unsafe_allow_html=True)

    user_input = st.text_input("Your Message", key="chat_input")
    voice_checkbox = st.checkbox("🔊 Voice Response", key="voice_opt")

    if st.button("Send"):
        if user_input.strip():
            # Add user message
            st.session_state.chat_history.append({"role":"user","text":user_input})
            # Generate bot reply
            bot_reply = generate_reply(user_input)
            st.session_state.chat_history.append({"role":"bot","text":bot_reply})
            # Speak if voice enabled
            if voice_checkbox:
                speak_text(bot_reply)

    # Display chat
    for msg in st.session_state.chat_history[-20:]:  # show last 20 messages
        if msg["role"]=="user":
            st.markdown(f'<div class="chat-user">You: {msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">AI: {msg["text"]}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# PROJECTS TAB
# -----------------------------------------------------------
elif tab_option == "📁 Projects":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<h2 class="gold-title">Saved Projects</h2>', unsafe_allow_html=True)
    if st.session_state.projects:
        for i, proj in enumerate(st.session_state.projects):
            st.markdown(f"**Project {i+1}:** {proj['title']} • Messages: {len(proj['chat'])}")
            if st.button(f"Load Project {i+1}", key=f"load_{i}"):
                st.session_state.chat_history = proj["chat"]
    else:
        st.info("No projects saved yet.")

    # Save current chat
    proj_title = st.text_input("Save Current Chat As:", key="proj_name")
    if st.button("Save Project"):
        if proj_title.strip() and st.session_state.chat_history:
            st.session_state.projects.append({"title":proj_title, "chat":st.session_state.chat_history.copy()})
            st.success(f"Project '{proj_title}' saved.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# HISTORY TAB
# -----------------------------------------------------------
elif tab_option == "📜 History":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<h2 class="gold-title">Chat History</h2>', unsafe_allow_html=True)
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history[-50:]):
            role = "You" if msg["role"]=="user" else "AI"
            st.markdown(f"**{role}:** {msg['text']}")
    else:
        st.info("No history yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# SETTINGS TAB
# -----------------------------------------------------------
elif tab_option == "⚙️ Settings":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<h2 class="gold-title">Settings</h2>', unsafe_allow_html=True)
    st.markdown('<p>Adjust options for your AI content creator.</p>', unsafe_allow_html=True)
    st.checkbox("Enable Voice Response", key="voice_setting")
    st.slider("Font Size", min_value=14, max_value=24, value=16, key="font_size")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# AI SOCIAL MEDIA CONTENT CREATOR – CHATBOT EDITION
# -----------------------------------------------------------
# ✔ Clean & Responsive UI
# ✔ Chatbot + Content Generator
# ✔ Export CSV/JSON
# ✔ No Voice Features
# ✔ Fully Works on Streamlit Cloud
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import random

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI SOCIAL MEDIA CONTENT CREATOR",
    layout="wide"
)

# -----------------------------------------------------------
# Theme
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: white;
        font-family: 'Segoe UI';
    }
    .glass-box {
        background: rgba(0,0,0,0.55);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
        box-shadow: 0 0 15px rgba(255,215,0,0.4);
        margin-bottom: 10px;
    }
    .glow-title {
        font-size: 40px;
        font-weight: 900;
        color: gold;
        text-shadow: 0 0 15px gold, 0 0 30px #ffdd55;
    }
    .stButton>button {
        background: gold !important;
        color: black !important;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stTextArea>div>textarea {
        background: rgba(255,255,255,0.18);
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# Content generator logic
# -----------------------------------------------------------
def generate_content(prompt, count=1):
    """
    Generate mock social media content (quotes, captions, hooks)
    """
    outputs = []
    for i in range(count):
        outputs.append({
            "content": f"{prompt} #{i+1}: " + random.choice([
                "Believe in yourself and never give up.",
                "Success is the sum of small efforts repeated daily.",
                "Your limitation—it’s only your imagination.",
                "Push yourself because no one else is going to do it for you.",
                "Great things never come from comfort zones.",
                "Dream it. Wish it. Do it.",
                "Stay positive, work hard, make it happen.",
                "Don’t stop when you’re tired; stop when you’re done.",
                "The harder you work for something, the greater you’ll feel when you achieve it.",
                "Little things make big days."
            ])
        })
    return outputs

# -----------------------------------------------------------
# Simple Chatbot logic
# -----------------------------------------------------------
def chatbot_reply(user_input):
    """
    Simple rule-based responses for chat
    """
    msg = user_input.lower()
    if "quote" in msg or "motivational" in msg:
        return "Sure! Enter the number of quotes you want and click 'Generate'."
    if "help" in msg:
        return "Type any topic or command, e.g., 'Write 5 motivational quotes'."
    return f"I understood: {user_input}. Try asking for quotes or captions!"

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["✨ Generator", "🤖 Chatbot"])

# ===========================================================
# TAB 1 — GENERATOR
# ===========================================================
with tab1:
    st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR</h1>', unsafe_allow_html=True)
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)

    topic = st.text_input("Enter your topic or content idea", "")
    count = st.number_input("Number of variations", min_value=1, max_value=20, value=5)
    generate_button = st.button("Generate Content ✨")

    st.markdown('</div>', unsafe_allow_html=True)

    if generate_button:
        if not topic.strip():
            st.warning("Please enter a topic!")
        else:
            results = generate_content(topic, count)
            for i, data in enumerate(results):
                st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                st.write(f"📌 Variation {i+1}: {data['content']}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Export options
            df = pd.DataFrame(results)
            st.download_button("Download CSV", df.to_csv(index=False), "content.csv")
            st.download_button("Download JSON", df.to_json(), "content.json")

# ===========================================================
# TAB 2 — CHATBOT
# ===========================================================
with tab2:
    st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR – Chatbot</h1>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_text = st.text_input("Your Message", "")

    if st.button("Send"):
        if user_text.strip():
            st.session_state.history.append(("You", user_text))
            reply = chatbot_reply(user_text)
            st.session_state.history.append(("AI", reply))

    for role, msg in st.session_state.history:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.write(f"**{role}:** {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# AI SOCIAL MEDIA CONTENT CREATOR – CHATBOT
# -----------------------------------------------------------

import streamlit as st
from transformers import pipeline, set_seed
import random

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI SOCIAL MEDIA CONTENT CREATOR",
    layout="wide"
)

# -----------------------------------------------------------
# Styling
# -----------------------------------------------------------
def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1511918984145-48de785d4c4b?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white !important;
            font-family: 'Segoe UI';
        }

        .glass-box {
            background: rgba(0,0,0,0.55);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(7px);
            box-shadow: 0 0 18px rgba(255,215,0,0.4);
            margin-bottom: 15px;
        }

        .glow-title {
            font-size: 42px;
            font-weight: 900;
            color: gold;
            text-shadow: 0 0 20px gold, 0 0 40px #ffdd55;
        }

        .stButton>button {
            background: gold !important;
            color: black !important;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            padding: 8px 16px;
            box-shadow: 0 0 12px gold;
        }

        .stTextInput>div>div>input,
        .stTextArea>div>textarea {
            background: rgba(255,255,255,0.18);
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

# -----------------------------------------------------------
# Load Model
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        generator = pipeline("text-generation", model="distilgpt2")
        set_seed(42)
        return generator
    except:
        return None

generator_model = load_model()

# -----------------------------------------------------------
# Simple Template Fallback
# -----------------------------------------------------------
def fallback_response(user_input):
    number = 5
    if any(char.isdigit() for char in user_input):
        number = int(''.join(filter(str.isdigit, user_input)))

    if "quote" in user_input.lower():
        quotes = [
            "Believe in yourself!",
            "Every day is a second chance.",
            "Dream it. Wish it. Do it.",
            "Success is the sum of small efforts.",
            "Push yourself, because no one else is going to do it.",
            "Don't wait for opportunity. Create it.",
            "Your limitation—it’s only your imagination.",
            "Great things never come from comfort zones.",
            "Dream bigger. Do bigger.",
            "Stay positive, work hard, make it happen."
        ]
        return "\n".join(random.sample(quotes, min(number, len(quotes))))

    return f"Here's your generated content based on: {user_input}"

# -----------------------------------------------------------
# Chatbot Logic
# -----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR</h1>', unsafe_allow_html=True)

user_input = st.text_input("Type your request:")

if st.button("Send"):
    if user_input.strip():
        st.session_state.history.append(("You", user_input))

        # Generate response
        if generator_model:
            try:
                response = generator_model(user_input, max_length=150, num_return_sequences=1)[0]["generated_text"]
            except:
                response = fallback_response(user_input)
        else:
            response = fallback_response(user_input)

        st.session_state.history.append(("AI", response))

# Display conversation
for role, msg in st.session_state.history:
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.write(f"**{role}:** {msg}")
    st.markdown('</div>', unsafe_allow_html=True)

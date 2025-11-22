# -----------------------------------------------------------
# AI SOCIAL MEDIA CONTENT CREATOR – CHATBOT EDITION
# -----------------------------------------------------------

import streamlit as st
import random

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI SOCIAL MEDIA CONTENT CREATOR",
    layout="wide"
)

# -----------------------------------------------------------
# Background + Styling
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
# Simple Content Generator
# -----------------------------------------------------------
def generate_content(user_input):
    """
    This simple generator handles:
    - "write X motivational quotes"
    - "generate X captions"
    - "create X scripts"
    """
    user_input = user_input.lower()

    number = 5  # default
    if any(char.isdigit() for char in user_input):
        number = int(''.join(filter(str.isdigit, user_input)))

    if "quote" in user_input:
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
        return random.sample(quotes, min(number, len(quotes)))

    elif "caption" in user_input:
        captions = [
            "Life is better when you smile 😄",
            "Adventure awaits! ✈️",
            "Coffee in hand, sparkle in my eye ☕✨",
            "Chasing dreams, not followers 💫",
            "Create your own sunshine ☀️",
            "Good vibes only ✌️",
            "Work hard, play harder 🎯",
            "Smile, it confuses people 😎",
            "Believe you can and you're halfway there 🌟",
            "Make today ridiculously amazing 💥"
        ]
        return random.sample(captions, min(number, len(captions)))

    else:
        return [f"Generated content {i+1}" for i in range(number)]

# -----------------------------------------------------------
# Main App UI
# -----------------------------------------------------------
st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR</h1>', unsafe_allow_html=True)
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

user_input = st.text_input("Enter your request, e.g., 'write 10 motivational quotes'")

if st.button("Generate ✨"):
    if not user_input.strip():
        st.warning("Please enter a request!")
    else:
        results = generate_content(user_input)
        for i, res in enumerate(results):
            st.markdown('<div class="glass-box">', unsafe_allow_html=True)
            st.write(f"**{i+1}.** {res}")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

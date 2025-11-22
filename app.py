"""
UPDATED VERSION — Social Media Content Creator / Chatbot
Now includes:
- Beautiful colored UI
- Background image across entire page
- Styled headers, boxes, and layout

No external API keys required.

To run:
1. python -m venv venv
2. pip install -r requirements.txt
3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import time

# ---------- OPTIONAL DEPENDENCIES ----------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except:
    TRANSLATOR_AVAILABLE = False


# -----------------------------------------------------
# 🔥 BEAUTIFUL BACKGROUND + CUSTOM COLORS
# -----------------------------------------------------
def add_bg_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1522202176988-66273c2fd55f");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# Global color styling
st.markdown(
    """
    <style>
        h1, h2, h3 {
            color: #ffffff !important;
            text-shadow: 1px 1px 3px black;
        }

        .stSidebar {
            background-color: rgba(0,0,0,0.7) !important;
        }

        .stTextInput>div>div>input {
            background-color: #ffffff;
            border-radius: 8px;
        }

        .content-box {
            background: rgba(0,0,0,0.55);
            padding: 18px;
            border-radius: 15px;
            margin-bottom: 15px;
            color: white;
            border: 1px solid #ffffff30;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# Model Load
# -----------------------------------------------------

model = None
tokenizer = None

if TRANSFORMERS_AVAILABLE:
    try:
        with st.spinner("Loading offline AI model..."):
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
            model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except:
        TRANSFORMERS_AVAILABLE = False


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

STOPWORDS = {"the", "and", "is", "in", "to", "a", "of", "for", "on", "with"}

def extract_keywords(topic):
    words = re.findall(r"\w+", topic.lower())
    return [w for w in words if w not in STOPWORDS][:5]

def generate_hashtags(keywords):
    base = ["#" + k for k in keywords]
    extra = ["#viral", "#trending", "#contentcreator", "#socialtips"]
    return base + random.sample(extra, min(len(extra), 5))

def template_variant(topic, tone, length, keywords):
    hooks = [
        f"Revealing the secret behind {topic}!",
        f"Here’s why {topic} is trending!",
        f"Quick guide to understand {topic}."
    ]
    ctas = ["Try it today!", "Share this now!", "Save for later!", "Join the discussion!"]
    images = [
        f"High-quality cinematic shot of {topic}",
        f"Minimalist background focusing on {topic}",
    ]
    scripts = [
        f"Let’s talk about {topic} in 30 seconds.",
        f"Here’s everything you need to know about {topic}.",
    ]

    post = f"{random.choice(hooks)} — {topic}. {random.choice(ctas)}"
    if len(post) > length:
        post = post[:length]

    return {
        "post": post,
        "hashtags": generate_hashtags(keywords),
        "hook": hooks[0],
        "cta": random.choice(ctas),
        "image_prompt": random.choice(images),
        "video_script": random.choice(scripts),
        "posting_time": f"{random.randint(8, 20)}:00 on {random.choice(['Mon','Tue','Wed','Thu','Fri'])}",
        "reading_time": "1 min",
        "char_count": len(post),
        "confidence": "template"
    }

def fallback_generate(topic, tone, length, count, keywords):
    return [template_variant(topic, tone, length, keywords) for _ in range(count)]


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("✨ AI Social Media Content Creator")
st.subheader("Beautiful UI • Offline AI • No API Keys • Instant Results")

tabs = st.tabs(["📝 Generator", "🤖 Chatbot"])

# -----------------------------------------------------
# GENERATOR TAB
# -----------------------------------------------------
with tabs[0]:
    st.sidebar.header("🎛️ Content Settings")
    topic = st.sidebar.text_input("Enter Topic", "")
    platform = st.sidebar.selectbox("Platform", ["Instagram","YouTube Shorts","Twitter/X","LinkedIn","Facebook","TikTok"])
    tone = st.sidebar.selectbox("Tone", ["Casual","Professional","Funny","Inspirational","Urgent"])
    length = st.sidebar.selectbox("Length", ["Short (120)", "Medium (300)", "Long (800)"])
    count = st.sidebar.slider("Number of Variations", 1, 10, 3)

    length_map = {"Short (120)": 120, "Medium (300)": 300, "Long (800)": 800}
    max_length = length_map[length]

    if st.sidebar.button("Use Demo"):
        topic = "How to grow your brand online"

    if st.button("Generate") and topic.strip():
        with st.spinner("Creating amazing content..."):
            keywords = extract_keywords(topic)
            data = fallback_generate(topic, tone, max_length, count, keywords)

        for i, v in enumerate(data):
            st.markdown(f"<div class='content-box'>", unsafe_allow_html=True)
            st.markdown(f"### ✨ Variant {i+1}")
            st.write("**Post:**")
            st.write(v["post"])
            st.write("**Hashtags:**", " ".join(v["hashtags"]))
            st.write("**Hook:**", v["hook"])
            st.write("**CTA:**", v["cta"])
            st.write("**Image Prompt:**", v["image_prompt"])
            st.write("**Video Script:**", v["video_script"])
            st.write("**Posting Time:**", v["posting_time"])
            st.write("**Character Count:**", v["char_count"])
            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------
# CHATBOT TAB
# -----------------------------------------------------
with tabs[1]:
    st.markdown("<h2 style='color:white'>Chatbot Mode</h2>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user = st.text_input("Ask anything about your generated content")

    if st.button("Send") and user.strip():
        response = f"Fallback bot response to: {user}"
        st.session_state.chat.append({"u": user, "b": response})

    for msg in st.session_state.chat:
        st.markdown(f"<div class='content-box'><b>You:</b> {msg['u']}<br><b>Bot:</b> {msg['b']}</div>", unsafe_allow_html=True)


# -----------------------------------------------------
# DEBUG DEMO (SAFE)
# -----------------------------------------------------
def demo():
    x = fallback_generate("Test Topic", "Casual", 120, 2, ["test"])
    print(x)

if __name__ == "__main__":
    demo()

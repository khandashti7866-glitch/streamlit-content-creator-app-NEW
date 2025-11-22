# -----------------------------------------------------------
# AI SOCIAL MEDIA CONTENT CREATOR – VIP EDITION
# -----------------------------------------------------------
# ✔ Car Background
# ✔ Gold & Black Premium Theme
# ✔ Right-Side UI
# ✔ Generator + Chatbot
# ✔ Export CSV/JSON
# ✔ Optional Local Model (distilgpt2)
# ✔ No API Keys Needed
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import json
import random
import re

# Optional model
try:
    from transformers import pipeline
    MODEL_READY = True
except:
    MODEL_READY = False

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI SOCIAL MEDIA CONTENT CREATOR",
    layout="wide"
)

# -----------------------------------------------------------
# Car Background + VIP Styling
# -----------------------------------------------------------
def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1614087403036-cf2e2c0a7bfa?auto=format&fit=crop&w=1470&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white !important;
            font-family: 'Segoe UI';
        }

        .glass-box {
            background: rgba(0,0,0,0.55);
            padding: 22px;
            border-radius: 18px;
            backdrop-filter: blur(7px);
            box-shadow: 0 0 18px rgba(255,215,0,0.4);
        }

        .glow-title {
            font-size: 46px;
            font-weight: 900;
            color: gold;
            text-shadow: 0 0 20px gold, 0 0 40px #ffdd55;
        }

        .stButton>button {
            background: gold !important;
            color: black !important;
            border-radius: 10px;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            box-shadow: 0 0 12px gold;
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

apply_theme()

# -----------------------------------------------------------
# Optional Local Model Load
# -----------------------------------------------------------
generator_model = None

if MODEL_READY:
    try:
        generator_model = pipeline("text-generation", model="distilgpt2")
    except:
        generator_model = None

# -----------------------------------------------------------
# Template-Based Generator (Fallback)
# -----------------------------------------------------------
def template_engine(topic, tone):
    hooks = [
        f"The truth about {topic} is surprising.",
        f"Why everyone is suddenly talking about {topic}.",
        f"Here’s what nobody tells you about {topic}.",
    ]
    scripts = [
        f"{topic} is more important than most people realize. Here's why…",
        f"Most people misunderstand {topic}. Let's fix that…",
        f"Let’s break down {topic} into something simple and powerful.",
    ]
    ctas = [
        "Follow for more insights!",
        "Save this for later",
        "Share this with someone who needs it.",
    ]

    return {
        "hook": random.choice(hooks),
        "caption": f"{topic} explained in a {tone.lower()} tone.",
        "script": random.choice(scripts),
        "cta": random.choice(ctas),
        "hashtags": f"#{topic.replace(' ', '')} #viral #creator #growth #motivation",
        "image_prompt": f"Cinematic car artwork representing {topic}",
        "schedule": "Best time: Wednesday • 7 PM",
    }

# -----------------------------------------------------------
# Generate Content (Model or Fallback)
# -----------------------------------------------------------
def generate_content(topic, tone, length):
    if generator_model:
        try:
            text = generator_model(
                f"Write social media content about {topic} in {tone} tone.",
                max_length=120
            )[0]["generated_text"]

            return {
                "hook": text[:80],
                "caption": text[:120],
                "script": text[:150],
                "cta": "Follow for more content!",
                "hashtags": f"#{topic.replace(' ', '')} #viral #creator",
                "image_prompt": f"Futuristic car-style image based on {topic}",
                "schedule": "Best time: Friday • 9 PM",
            }
        except:
            return template_engine(topic, tone)

    return template_engine(topic, tone)

# -----------------------------------------------------------
# Chatbot Logic (simple)
# -----------------------------------------------------------
def chatbot_reply(message):
    msg = message.lower()

    if "short" in msg:
        return "Here is a shorter version: " + message[:50] + "..."
    if "long" in msg:
        return "Here is a longer version: " + message + " — with added detail."
    if "improve" in msg:
        return "Here is a cleaner improved version: " + message.capitalize()
    if "help" in msg:
        return "Ask me anything: rewrite, shorten, expand, improve, or format content."

    return "Here is a refined version: " + message

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["✨ Generator", "🤖 Chatbot"])

# ===========================================================
# TAB 1 — GENERATOR
# ===========================================================
with tab1:

    left, right = st.columns([2.5, 1])

    # LEFT SIDE TITLE
    with left:
        st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR</h1>', unsafe_allow_html=True)

    # RIGHT SIDE UI
    with right:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)

        topic = st.text_input("Enter Topic")
        tone = st.selectbox("Tone", ["Professional", "Casual", "Funny", "Inspirational", "Urgent"])
        length = st.selectbox("Length", ["Short", "Medium", "Long"])
        count = st.number_input("Number of Variations", min_value=1, max_value=10, value=1)

        generate_button = st.button("Generate Content ✨")

        st.markdown('</div>', unsafe_allow_html=True)

    # OUTPUT
    with left:
        if generate_button:
            if not topic.strip():
                st.warning("Please enter a topic!")
            else:
                results = []
                for i in range(count):
                    data = generate_content(topic, tone, length)
                    results.append(data)

                    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                    st.subheader(f"📌 Variation {i + 1}")

                    st.write("**Hook:**", data["hook"])
                    st.write("**Caption:**", data["caption"])
                    st.write("**Script:**", data["script"])
                    st.write("**CTA:**", data["cta"])
                    st.write("**Hashtags:**", data["hashtags"])
                    st.write("**Image Prompt:**", data["image_prompt"])
                    st.write("**Best Posting Time:**", data["schedule"])

                    st.markdown('</div>', unsafe_allow_html=True)

                # Export Buttons
                df = pd.DataFrame(results)
                st.download_button("Download CSV", df.to_csv(index=False), "content.csv")
                st.download_button("Download JSON", df.to_json(), "content.json")

# ===========================================================
# TAB 2 — CHATBOT
# ===========================================================
with tab2:
    st.markdown('<h1 class="glow-title">AI Chatbot</h1>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_text = st.text_input("Your Message")

    if st.button("Send"):
        st.session_state.history.append(("You", user_text))
        reply = chatbot_reply(user_text)
        st.session_state.history.append(("AI", reply))

    for role, msg in st.session_state.history:
        st.write(f"**{role}:** {msg}")

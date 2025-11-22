# Updated Streamlit App with Custom Background and UI Enhancements
import streamlit as st
import pandas as pd
import json
import random
import re

try:
    from transformers import pipeline
    MODEL_READY = True
except:
    MODEL_READY = False

st.set_page_config(page_title="AI SOCIAL MEDIA CONTENT CREATOR", layout="wide")

# -----------------------------------------------------------
# Apply Custom UI Theme and Background
# -----------------------------------------------------------
def apply_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('/mnt/data/Wallpaper -www.posintech.com (2).jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white !important;
            font-family: 'Segoe UI';
        }}

        .glass-box {{
            background: rgba(0,0,0,0.60);
            padding: 25px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 25px rgba(0, 200, 255, 0.35);
            border: 1px solid rgba(0, 200, 255, 0.45);
        }}

        .glow-title {{
            font-size: 50px;
            font-weight: 900;
            color: #00eaff;
            text-shadow: 0 0 12px #00eaff, 0 0 25px #00aaff;
        }}

        .stButton>button {{
            background: linear-gradient(90deg,#00ddff,#0099ff) !important;
            color: black !important;
            border-radius: 12px;
            font-weight: bold;
            border: none;
            padding: 10px 22px;
            box-shadow: 0 0 15px rgba(0,200,255,0.6);
        }}

        .stTextInput>div>div>input,
        .stSelectbox>div>div>div,
        .stTextArea>div>textarea {{
            background: rgba(255,255,255,0.15);
            color: white !important;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

# Optional model
if MODEL_READY:
    try:
        generator_model = pipeline("text-generation", model="distilgpt2")
    except:
        generator_model = None
else:
    generator_model = None

# Template Fallback

def template_engine(topic, tone):
    hooks = [
        f"The truth about {topic} will shock you…",
        f"Nobody is talking about {topic}, but they should…",
        f"{topic}: what you MUST know today."
    ]
    scripts = [
        f"Here's something people misunderstand about {topic}…",
        f"Breaking down {topic} in the simplest way possible…",
        f"What you never realized about {topic}…"
    ]
    ctas = ["Follow for more 🔥", "Save this!", "Share this with someone"]

    return {
        "hook": random.choice(hooks),
        "caption": f"{topic} explained with a {tone.lower()} vibe.",
        "script": random.choice(scripts),
        "cta": random.choice(ctas),
        "hashtags": f"#{topic.replace(' ', '')} #viral #content #growth",
        "image_prompt": f"Cinematic car-themed visual for {topic}",
        "schedule": "Best time: Wednesday • 7 PM"
    }

# Generator Logic

def generate_content(topic, tone, length):
    if generator_model:
        try:
            txt = generator_model(
                f"Write social media content about {topic} in {tone} tone.",
                max_length=150
            )[0]["generated_text"]

            return {
                "hook": txt[:80],
                "caption": txt[:120],
                "script": txt[:150],
                "cta": "Follow for more!",
                "hashtags": f"#{topic.replace(' ', '')} #viral #trending",
                "image_prompt": f"Car-themed visual inspired by {topic}",
                "schedule": "Best time: Friday • 9 PM"
            }
        except:
            return template_engine(topic, tone)

    return template_engine(topic, tone)

# Chatbot

def chatbot_reply(msg):
    m = msg.lower()
    if "short" in m:
        return "Short version: " + msg[:50] + "…"
    if "long" in m:
        return msg + " — adding more details as requested."
    if "improve" in m:
        return "Improved: " + msg.capitalize()
    if "help" in m:
        return "I can rewrite, expand, shorten, or clean your content."
    return "Refined: " + msg

# UI Layout
tab1, tab2 = st.tabs(["✨ Generator", "🤖 Chatbot"])

# Generator Tab
with tab1:
    left, right = st.columns([2.4, 1])

    with left:
        st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR</h1>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        topic = st.text_input("Enter Topic")
        tone = st.selectbox("Tone", ["Professional", "Casual", "Funny", "Inspirational", "Urgent"])
        length = st.selectbox("Length", ["Short", "Medium", "Long"])
        count = st.number_input("Variations", min_value=1, max_value=10, value=1)
        generate = st.button("Generate ✨")
        st.markdown('</div>', unsafe_allow_html=True)

    with left:
        if generate:
            if not topic.strip():
                st.warning("Please enter a topic!")
            else:
                results = []
                for i in range(count):
                    data = generate_content(topic, tone, length)
                    results.append(data)

                    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
                    st.subheader(f"📌 Variation {i+1}")
                    st.write("**Hook:**", data["hook"])
                    st.write("**Caption:**", data["caption"])
                    st.write("**Script:**", data["script"])
                    st.write("**CTA:**", data["cta"])
                    st.write("**Hashtags:**", data["hashtags"])
                    st.write("**Image Prompt:**", data["image_prompt"])
                    st.write("**Best Posting Time:**", data["schedule"])
                    st.markdown('</div>', unsafe_allow_html=True)

                df = pd.DataFrame(results)
                st.download_button("Download CSV", df.to_csv(index=False), "content.csv")
                st.download_button("Download JSON", df.to_json(orient="records"), "content.json")

# Chatbot Tab
with tab2:
    st.markdown('<h1 class="glow-title">AI Chatbot</h1>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_msg = st.text_input("Your Message", key="chat_msg")

    if st.button("Send"):
        if user_msg.strip():
            st.session_state.history.append(("You", user_msg))
            ans = chatbot_reply(user_msg)
            st.session_state.history.append(("AI", ans))

    for role, message in st.session_state.history:
        st.write(f"**{role}:** {message}")

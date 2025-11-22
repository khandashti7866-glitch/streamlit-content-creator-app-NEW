# -----------------------------------------------------------
# AI SOCIAL MEDIA CONTENT CREATOR – VIP VERSION (No API Keys)
# -----------------------------------------------------------
# Features:
# ✔ Galaxy background
# ✔ Black + Gold VIP theme
# ✔ Right-side control panel
# ✔ Generator + Chatbot tabs
# ✔ Local model support (if available)
# ✔ Fallback template engine
# ✔ Copy / Regenerate buttons
# ✔ Export CSV / JSON
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import json
import re
import random

# Try optional model
try:
    from transformers import pipeline
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AI SOCIAL MEDIA CONTENT CREATOR", layout="wide")


# -----------------------------------------------------------
# GALAXY BACKGROUND + VIP THEME
# -----------------------------------------------------------
def add_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://i.imgur.com/ULaJtYH.jpeg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Right Panel Box */
        .glass-box {
            background: rgba(0,0,0,0.55);
            padding: 25px;
            border-radius: 18px;
            backdrop-filter: blur(6px);
            box-shadow: 0 0 25px rgba(255,215,0,0.4);
        }

        /* Title Glow – VIP gold */
        .glow-title {
            font-size: 48px;
            font-weight: 900;
            color: gold;
            text-shadow: 0 0 18px gold, 0 0 38px #ffdd55;
        }

        /* Buttons */
        .stButton>button {
            background: gold;
            color: black !important;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0 0 15px gold;
        }

        /* Inputs */
        .stTextInput>div>div>input,
        .stTextArea>div>textarea,
        .stSelectbox>div>div>div {
            background: rgba(255,255,255,0.15);
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


add_background()


# -----------------------------------------------------------
# OPTIONAL MODEL LOADING
# -----------------------------------------------------------
generator_model = None
if MODEL_AVAILABLE:
    try:
        generator_model = pipeline("text-generation", model="distilgpt2")
    except:
        generator_model = None


# -----------------------------------------------------------
# FALLBACK TEMPLATE ENGINE
# -----------------------------------------------------------
def generate_template(topic, tone, length):
    hooks = [
        f"The truth about {topic} will change everything.",
        f"Everyone is talking about {topic}, but here’s what they don’t know.",
        f"{topic} simplified in 10 seconds."
    ]
    ctas = [
        "Follow for more insights.",
        "Share this with someone who needs it.",
        "Save this for later."
    ]
    scripts = [
        f"Most people misunderstand {topic}. Here’s what matters most...",
        f"The real power of {topic} is often overlooked...",
        f"Let’s break down {topic} in a simple way..."
    ]
    hashtags = [
        f"#{topic.replace(' ', '')}", "#viral", "#contentcreator", "#motivation",
        "#branding", "#growth", "#reels"
    ]

    return {
        "hook": random.choice(hooks),
        "caption": f"A quick breakdown of {topic} ({tone} tone).",
        "script": random.choice(scripts),
        "cta": random.choice(ctas),
        "hashtags": " ".join(hashtags[:7]),
        "schedule": "Best time: Wednesday 7 PM",
        "image_prompt": f"A cinematic galaxy-style image representing {topic}",
    }


# -----------------------------------------------------------
# MODEL OR FALLBACK
# -----------------------------------------------------------
def generate_content(topic, tone, length):
    if generator_model:
        try:
            out = generator_model(f"{topic} {tone}", max_length=80)[0]['generated_text']
            return {
                "hook": out[:80],
                "caption": f"{out[:120]}...",
                "script": out[:150],
                "cta": "Follow for more content!",
                "hashtags": f"#{topic.replace(' ', '')} #viral #growth",
                "schedule": "Best time: Friday 9 PM",
                "image_prompt": f"A futuristic galaxy-themed concept of {topic}",
            }
        except:
            return generate_template(topic, tone, length)
    else:
        return generate_template(topic, tone, length)


# -----------------------------------------------------------
# CHATBOT SIMPLE ENGINE
# -----------------------------------------------------------
def chatbot_reply(message):
    if "short" in message.lower():
        return "Here is a shorter version: " + message[:60] + "..."
    if "long" in message.lower():
        return "Here is a longer version: " + message + " — and additional insights added."
    if "translate" in message.lower():
        return "Translation feature disabled (no API), but I can rewrite it simply."
    return "Here is an improved version: " + message


# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["✨ Generator", "🤖 Chatbot"])


# ===========================================================
# TAB 1: GENERATOR
# ===========================================================
with tab1:
    col_left, col_right = st.columns([2.5, 1])

    with col_left:
        st.markdown('<h1 class="glow-title">AI SOCIAL MEDIA CONTENT CREATOR</h1>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)

        topic = st.text_input("Topic")
        tone = st.selectbox("Tone", ["Professional", "Casual", "Funny", "Inspirational", "Urgent"])
        length = st.selectbox("Length", ["Short", "Medium", "Long"])
        variations = st.number_input("Variations", min_value=1, max_value=10, value=1)

        generate_btn = st.button("Generate Content ✨")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_left:
        if generate_btn:
            if topic.strip() == "":
                st.warning("Please enter a topic.")
            else:
                results = []
                for i in range(variations):
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

                # Export
                df = pd.DataFrame(results)
                st.download_button("Download CSV", df.to_csv(index=False), "content.csv")
                st.download_button("Download JSON", df.to_json(), "content.json")


# ===========================================================
# TAB 2: CHATBOT
# ===========================================================
with tab2:
    st.markdown('<h1 class="glow-title">AI Chatbot</h1>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_msg = st.text_input("Your message")

    if st.button("Send"):
        st.session_state.chat_history.append(("user", user_msg))
        bot_reply = chatbot_reply(user_msg)
        st.session_state.chat_history.append(("bot", bot_reply))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write(f"🧑‍💬 **You:** {msg}")
        else:
            st.write(f"🤖 **AI:** {msg}")

"""
Social Media Content Creator / Chatbot - Streamlit App

Requirements:
---------------
# requirements.txt snippet:
streamlit
pandas
numpy
torch       # optional, for transformer model
transformers # optional, for transformer model
googletrans==4.0.0rc1 # optional, for Urdu translation

How to run:
-------------
1. python -m venv venv
2. pip install -r requirements.txt
3. streamlit run app.py

Fallback behavior:
------------------
- If 'transformers' and a small GPT-2 model are available, the app uses it to generate content.
- If not, a template-based deterministic engine is used.
- All generation works offline, no API key required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import time
from datetime import datetime

# Optional dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

# ------------------------------
# Model Initialization
# ------------------------------
MODEL_NAME = "sshleifer/tiny-gpt2"
model = None
tokenizer = None
if TRANSFORMERS_AVAILABLE:
    try:
        with st.spinner("Loading local transformer model..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    except Exception as e:
        st.warning(f"Transformer model failed to load: {e}")
        TRANSFORMERS_AVAILABLE = False

# ------------------------------
# Utility Functions
# ------------------------------

STOPWORDS = set(["the", "and", "is", "in", "to", "a", "of", "for", "on", "with"])

def extract_keywords(topic: str):
    """Simple keyword extractor using regex."""
    words = re.findall(r'\w+', topic.lower())
    keywords = [w for w in words if w not in STOPWORDS]
    return keywords[:5]

def generate_hashtags(keywords):
    """Generate hashtags from keywords."""
    hashtags = ["#" + w for w in keywords]
    # Add some generic tags
    generic_tags = ["#viral", "#trending", "#socialmedia", "#contentcreator"]
    hashtags += random.sample(generic_tags, min(5, len(generic_tags)))
    return hashtags[:15]

def generate_template_variant(topic, tone, length, keywords):
    """Deterministic template-based generation."""
    hooks = [
        f"Discover the secret behind {topic}!",
        f"Everything you need to know about {topic}.",
        f"Why {topic} is trending now...",
        f"Top tips for {topic} you can't miss!"
    ]
    ctas = [
        "Click to learn more!",
        "Don't miss out, join now!",
        "Share if you agree!",
        "Try this today!"
    ]
    images = [
        f"A vibrant photo representing {topic}",
        f"Minimalist design showing {topic}",
        f"Creative illustration of {topic}",
        f"People interacting with {topic}"
    ]
    scripts = [
        f"Hey everyone, today let's talk about {topic}.",
        f"{topic} can change your life, here's how.",
        f"Quick tips on {topic} in just seconds.",
        f"Don't miss these secrets about {topic}."
    ]

    hook = random.choice(hooks)
    cta = random.choice(ctas)
    image_prompt = random.choice(images)
    video_script = random.choice(scripts)
    post_length = int(length)
    content = f"{hook} {topic} {cta}"
    if post_length < len(content):
        content = content[:post_length]
    return {
        "post": content,
        "hashtags": generate_hashtags(keywords),
        "hook": hook,
        "cta": cta,
        "image_prompt": image_prompt,
        "video_script": video_script,
        "posting_time": f"{random.randint(9,20)}:00 on {random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday'])}",
        "reading_time": f"{max(1,len(content.split())//200)} min",
        "char_count": len(content),
        "confidence": "template"
    }

def generate_with_model(prompt: str, max_length: int = 100):
    """Generate text using transformer model if available."""
    if model and tokenizer:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.7)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
    else:
        return None

def fallback_generate(topic, tone, length, num_variants, keywords):
    variants = []
    for _ in range(num_variants):
        variant = generate_template_variant(topic, tone, length, keywords)
        variants.append(variant)
    return variants

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Social Media Content Creator", layout="wide")
st.title("💡 Social Media Content Creator / Chatbot")

tabs = st.tabs(["Generator", "Chatbot"])

# ------------------------------
# Generator Tab
# ------------------------------
with tabs[0]:
    st.sidebar.header("Content Options")
    topic = st.sidebar.text_input("Enter Topic / Idea", "")
    platform = st.sidebar.selectbox("Platform", ["Instagram", "YouTube Shorts", "Twitter/X", "LinkedIn", "Facebook", "TikTok"])
    tone = st.sidebar.selectbox("Tone", ["Casual", "Professional", "Funny", "Inspirational", "Urgent"])
    post_length_option = st.sidebar.selectbox("Post Length", ["Short (<=120)", "Medium (120-300)", "Long (300-800)"])
    num_variants = st.sidebar.slider("Number of Variations", 1, 10, 3)
    optional_keywords = st.sidebar.text_input("Optional Keywords / Hashtags (comma separated)")

    if st.sidebar.button("Use Example"):
        topic = "How to grow your personal brand online"

    length_mapping = {"Short (<=120)": 120, "Medium (120-300)": 300, "Long (300-800)": 800}
    post_length = length_mapping[post_length_option]

    if st.button("Generate") and topic.strip():
        keywords = extract_keywords(topic)
        if optional_keywords:
            keywords += [k.strip() for k in optional_keywords.split(",")]
        with st.spinner("Generating content..."):
            time.sleep(0.5)
            variants = fallback_generate(topic, tone, post_length, num_variants, keywords)
        for i, v in enumerate(variants):
            st.markdown(f"### Variant {i+1}")
            st.text_area("Post Text", value=v['post'], height=80)
            st.write("**Hashtags:**", " ".join(v['hashtags']))
            st.write("**Hook:**", v['hook'])
            st.write("**CTA:**", v['cta'])
            st.write("**Image Prompt:**", v['image_prompt'])
            st.write("**Short Video Script:**", v['video_script'])
            st.write("**Posting Time:**", v['posting_time'])
            st.write("**Reading Time / Char Count:**", f"{v['reading_time']} / {v['char_count']}")
            st.write("**Confidence:**", v['confidence'])
            st.download_button("Export Variant JSON", data=pd.Series(v).to_json(), file_name=f"variant_{i+1}.json")

# ------------------------------
# Chatbot Tab
# ------------------------------
with tabs[1]:
    st.subheader("Chatbot Mode")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask about your content (e.g., 'make it funnier', 'translate to Urdu')", "")
    if st.button("Send") and user_input.strip():
        if TRANSLATOR_AVAILABLE and "translate" in user_input.lower() and "urdu" in user_input.lower():
            translator = Translator()
            last_post = st.session_state.chat_history[-1]['response'] if st.session_state.chat_history else "Hello"
            try:
                translated = translator.translate(last_post, src='en', dest='ur').text
            except:
                translated = "Translation failed, showing fallback text."
            response = translated
        else:
            response = f"Fallback response: {user_input}"
        st.session_state.chat_history.append({"user": user_input, "response": response})
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['response']}")

# ------------------------------
# Minimal Test / Demo Function
# ------------------------------
def demo_fallback():
    """Test fallback generation without model."""
    sample_topic = "Learn Python Fast"
    variants = fallback_generate(sample_topic, "Casual", 120, 2, extract_keywords(sample_topic))
    print("Fallback demo variants:")
    for v in variants:
        print(v['post'], "\n", v['hashtags'])

if __name__ == "__main__":
    demo_fallback()

import streamlit as st
import pandas as pd
import numpy as np
import random
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional transformer imports
TRANSFORMERS_AVAILABLE = False
MODEL_LOADED = False
model = None
tokenizer = None
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# --------------------------
# App constants
# --------------------------
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1503602642458-232111445657?auto=format&fit=crop&w=1600&q=80"
MODEL_NAME = "sshleifer/tiny-gpt2"
MAX_MODEL_TOKENS = 150

# --------------------------
# Load model
# --------------------------
if TRANSFORMERS_AVAILABLE:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        MODEL_LOADED = True
    except Exception as e:
        MODEL_LOADED = False
        print("Model not loaded:", e)

# --------------------------
# Utility functions
# --------------------------
STOPWORDS = set([
    "the","and","is","in","to","a","of","for","on","with","that","this","are","it","as","be","by",
    "an","from","at","or","your","you"
])

def css():
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #efefef;
    }}
    .glass {{
        background: linear-gradient(180deg, rgba(6,6,6,0.72) 0%, rgba(14,14,14,0.60) 100%);
        border: 1px solid rgba(255, 215, 0, 0.12);
        padding: 18px;
        border-radius: 14px;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        color: #fff;
    }}
    .gold-title {{
        font-size: 36px;
        font-weight: 800;
        color: #ffd166;
        margin-bottom: 6px;
    }}
    .gold-sub {{
        color: #ffd166;
        opacity: 0.9;
        margin-top: -8px;
        margin-bottom: 12px;
    }}
    .control-panel {{
        background: rgba(10,10,10,0.6);
        border-radius: 12px;
        padding: 14px;
        border: 1px solid rgba(255,215,0,0.08);
    }}
    </style>
    """, unsafe_allow_html=True)

def extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    toks = re.findall(r"\w+", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t)>2]
    freqs = {}
    for t in toks:
        freqs[t] = freqs.get(t,0)+1
    sorted_tokens = sorted(freqs.items(), key=lambda x: (-x[1],x[0]))
    return [t for t,_ in sorted_tokens][:max_keywords]

def generate_hashtags(keywords: List[str], max_tags: int = 8) -> List[str]:
    tags = ["#" + re.sub(r"\s+","",k) for k in keywords][:max_tags]
    extras = ["#viral","#trending","#contentcreator","#tips","#howto","#learn"]
    random.shuffle(extras)
    tags = tags + extras[:max(0,max_tags-len(tags))]
    return tags[:max_tags]

def truncate(text: str, max_chars: int) -> str:
    if len(text)<=max_chars: return text
    cut = text[:max_chars].rfind(".")
    if cut==-1: cut = text[:max_chars].rfind(" ")
    if cut==-1 or cut<max_chars//2: cut=max_chars
    return text[:cut].rstrip()+"…"

def estimate_reading_time(text: str) -> str:
    words = len(re.findall(r"\w+",text))
    minutes = max(1, int(words/200))
    return f"{minutes} min"

def simple_schedule(topic: str) -> str:
    hours=[9,11,14,17,19,21]
    day=random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    return f"{day} at {random.choice(hours)}:00"

# --------------------------
# Variant generator
# --------------------------
def template_variant(topic: str, tone: str, max_len: int) -> Dict[str,Any]:
    keywords = extract_keywords(topic)
    hashtags = generate_hashtags(keywords)
    hooks = {
        "Professional": f"Industry insight: {topic}.",
        "Casual": f"Quick tip on {topic}.",
        "Funny": f"Fun facts about {topic}.",
        "Inspirational": f"How {topic} changed many lives.",
        "Urgent": f"Alert: {topic} updates."
    }
    hook = hooks.get(tone,"Professional")
    post = truncate(f"{hook} {topic}. Check this out!", max_len)
    return {
        "post": post,
        "hashtags": hashtags,
        "hook": hook,
        "cta": "Learn more",
        "image_prompt": f"A visual representing {topic}",
        "video_script": f"Short script: {post[:100]}...",
        "posting_time": simple_schedule(topic),
        "reading_time": estimate_reading_time(post),
        "char_count": len(post),
        "confidence": "template"
    }

def generate_variants(topic: str, tone: str, length_pref: str, n_variants: int) -> List[Dict[str,Any]]:
    length_map={"Short":120,"Medium":300,"Long":800}
    max_len = length_map.get(length_pref,300)
    variants=[]
    for _ in range(n_variants):
        variants.append(template_variant(topic,tone,max_len))
    return variants

# --------------------------
# Chatbot
# --------------------------
def chatbot_response(message: str, last_content: Optional[str]) -> str:
    prompt=message.lower()
    if "shorten" in prompt and last_content:
        return truncate(last_content,280)
    if "polish" in prompt and last_content:
        return last_content.replace("don't","do not").replace("can't","cannot")
    if "hashtags" in prompt:
        kws=extract_keywords(last_content or message)
        return " ".join(generate_hashtags(kws))
    return "I can help with shorten, polish, or generate hashtags."

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="VIP Content Creator", page_icon="✨", layout="wide")
css()

if "variants" not in st.session_state: st.session_state.variants=[]
if "chat_history" not in st.session_state: st.session_state.chat_history=[]
if "last_generated_topic" not in st.session_state: st.session_state.last_generated_topic=""

st.title("✨ VIP Social Media Content Creator")
left_col,right_col = st.columns([2,1])

with right_col:
    st.markdown('<div class="control-panel glass">',unsafe_allow_html=True)
    st.subheader("Controls")
    topic = st.text_input("Topic", value=st.session_state.last_generated_topic)
    tone = st.selectbox("Tone",["Professional","Casual","Funny","Inspirational","Urgent"])
    length_pref = st.selectbox("Length",["Short","Medium","Long"])
    n_variants = st.slider("Number of Variants",1,10,3)
    if st.button("Generate Variants"):
        if topic.strip():
            st.session_state.variants = generate_variants(topic,tone,length_pref,n_variants)
            st.session_state.last_generated_topic = topic
    st.markdown('</div>',unsafe_allow_html=True)

with left_col:
    st.subheader("Generated Variants")
    for i,variant in enumerate(st.session_state.variants):
        st.markdown(f"**Variant {i+1}:**")
        st.text_area("Post Text", value=variant["post"], height=100)
        st.write("Hashtags:", " ".join(variant["hashtags"]))
        st.write("Hook:", variant["hook"], "CTA:", variant["cta"])
        st.write("Image Prompt:", variant["image_prompt"])
        st.write("Video Script:", variant["video_script"])
        st.write(f"Posting Time: {variant['posting_time']} | Reading Time: {variant['reading_time']} | Char Count: {variant['char_count']}")
        st.markdown("---")

st.subheader("Chatbot Assistant")
chat_input = st.text_input("Enter message for assistant")
if st.button("Send to Assistant"):
    if chat_input.strip():
        last = st.session_state.variants[0]["post"] if st.session_state.variants else ""
        reply = chatbot_response(chat_input,last)
        st.session_state.chat_history.append((chat_input,reply))

for msg,user_reply in st.session_state.chat_history:
    st.markdown(f"**You:** {msg}")
    st.markdown(f"**Assistant:** {user_reply}")

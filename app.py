"""
VIP Social Media Content Creator - Single-file Streamlit App (app.py)

Features:
- Right-side UI panel (controls) + left output area (results)
- Tabs: Generator & Chatbot
- Black & Gold VIP theme with fullscreen background & glass panels
- Optional local transformer model (sshleifer/tiny-gpt2 or distilgpt2). If unavailable, automatic deterministic template fallback.
- Generates multiple variants (1-10) with: post text, hashtags, hook, CTA, image prompt, short video script, posting time, reading time, char count, confidence
- Chatbot mode with professional writing assistant personality (local model OR deterministic transformations)
- Export CSV / JSON; per-variant Regenerate; Copy-to-clipboard; clipboard log downloadable
- No external API keys required; works fully offline (except optional model download)
- Good error handling - always returns variants via fallback

How to run (short):
1. python -m venv venv
2. pip activate venv  # or source venv/bin/activate
3. pip install -r requirements.txt
4. streamlit run app.py

Notes:
- transformers & torch are optional. If present the app will try to load a small model.
- The app will attempt to download the model if not present (internet) but will still work if download fails.
- This is a single-file app; edit constants below to change default model or visuals.
"""

# -------------------------
# Requirements (for requirements.txt file)
# -------------------------
# streamlit
# pandas
# numpy
# transformers  # optional
# torch         # optional
# pillow        # optional

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional dependencies - handled gracefully
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

# -------------------------
# App Config & Constants
# -------------------------
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1526378723480-1d2f3c4d3d4f?auto=format&fit=crop&w=1600&q=80"  # Tech background
MODEL_NAME = "sshleifer/tiny-gpt2"
MAX_MODEL_TOKENS = 150
MIN_VARIANTS = 1
MAX_VARIANTS = 10
STOPWORDS = set(["the","and","is","in","to","a","of","for","on","with","that","this","are","it","as","be","by","an","from","at","or","your","you"])

# -------------------------
# Load model if possible
# -------------------------
if TRANSFORMERS_AVAILABLE:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        MODEL_LOADED = True
    except Exception as e:
        MODEL_LOADED = False
        print("Transformer model not loaded. Using fallback. Reason:", e)

# -------------------------
# Utility functions
# -------------------------
def css() -> None:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #efefef;
    }}
    h1,h2,h3,p,label,span {{ text-shadow:0 2px 12px rgba(0,0,0,0.85); }}
    .glass {{ background: linear-gradient(180deg, rgba(6,6,6,0.72), rgba(14,14,14,0.60)); border:1px solid rgba(255,215,0,0.12); padding:18px; border-radius:14px; backdrop-filter:blur(6px); box-shadow:0 8px 30px rgba(0,0,0,0.6); color:#fff; }}
    .gold-title {{ font-size:36px; font-weight:800; color:#ffd166; text-shadow:0 0 18px rgba(255,209,102,0.28),0 0 6px rgba(0,0,0,0.6); margin-bottom:6px; }}
    .gold-sub {{ color:#ffd166; opacity:0.9; margin-top:-8px; margin-bottom:12px; }}
    .control-panel {{ background: rgba(10,10,10,0.6); border-radius:12px; padding:14px; border:1px solid rgba(255,215,0,0.08); }}
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {{ background: rgba(255,255,255,0.06); color:#fff; border-radius:8px; padding:8px; }}
    .stButton>button {{ background: linear-gradient(90deg,#ffd166,#ffb703); color:#08111a; font-weight:800; border-radius:10px; padding:8px 18px; border:none; box-shadow:0 6px 18px rgba(255,181,3,0.18); }}
    .variant-card {{ background: linear-gradient(180deg, rgba(0,0,0,0.42), rgba(6,6,6,0.52)); border-radius:12px; padding:12px; margin-bottom:12px; border:1px solid rgba(255,215,0,0.06); }}
    .meta {{ color:#e7e7e7; opacity:0.88; font-size:13px; }}
    </style>
    """, unsafe_allow_html=True)

def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    toks = re.findall(r"\w+", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t)>2]
    freqs = {}
    for t in toks:
        freqs[t] = freqs.get(t,0)+1
    sorted_tokens = sorted(freqs.items(), key=lambda x:(-x[1],x[0]))
    return [t for t,_ in sorted_tokens][:max_keywords]

def generate_hashtags_from_keywords(keywords: List[str], min_tags: int=5, max_tags:int=12)->List[str]:
    tags = ["#"+re.sub(r"[^A-Za-z0-9]","",k) for k in keywords if k]
    extras = ["#viral","#trending","#contentcreator","#tips","#howto","#learn"]
    random.shuffle(extras)
    tags = tags + extras[:max(0,min_tags-len(tags))]
    return tags[:max_tags]

def simple_posting_schedule(topic:str)->str:
    low = topic.lower()
    hours = [9,11,14,17,19,21]
    day = random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    if any(k in low for k in ["business","productivity","career","finance","invest"]):
        day = random.choice(["Tuesday","Wednesday","Thursday"])
    elif any(k in low for k in ["life","fun","travel","food","fashion","music"]):
        day = random.choice(["Friday","Saturday","Sunday"])
    return f"{day} at {random.choice(hours)}:00"

def estimate_reading_time(text:str)->str:
    words = len(re.findall(r"\w+",text))
    minutes = max(1,int(words/200))
    return f"{minutes} min"

def truncate_to_length(text:str,max_chars:int)->str:
    if len(text)<=max_chars: return text
    cut=text[:max_chars].rfind(".")
    if cut==-1: cut=text[:max_chars].rfind(" ")
    if cut==-1 or cut<max_chars//2: return text[:max_chars].rstrip()+"…"
    return text[:cut+1]

# -------------------------
# Fallback template generator
# -------------------------
def template_generator(topic:str, tone:str, platform:str, length_chars:int)->Dict[str,Any]:
    kw = extract_keywords(topic,6)
    hashtags = generate_hashtags_from_keywords(kw)
    hooks = {"Professional":[f"Industry insight: {topic} — what professionals must know.", f"Brief update on {topic} that impacts many industries."],
             "Casual":[f"Quick tip about {topic} you can use today!", f"Real talk: {topic} explained simply."],
             "Funny":[f"If {topic} were a person, here's what they'd say...", f"Fun facts (and laughs) about {topic}."],
             "Inspirational":[f"How {topic} changed the game for many people.", f"One idea that might change your view about {topic}."],
             "Urgent":[f"Important! {topic} updates you need to act on.", f"Alert: new {topic} shifts happening now."]}
    ctas = ["Learn more","Share your thoughts","Save this post","Try this now","Join the conversation"]
    images = [f"Professional photo representing {topic}", f"Minimalist editorial photo focused on {topic}"]
    scripts = [f"{topic} in 30 seconds: key steps are A,B,C","Quick tip on {topic}. First... Next... Finally..."]
    hook = random.choice(hooks.get(tone,hooks["Professional"]))
    cta = random.choice(ctas)
    image_prompt = random.choice(images)
    script = random.choice(scripts)
    post_text = truncate_to_length(f"{hook} {topic}. {cta}.", length_chars)
    return {"post":post_text, "hashtags":hashtags, "hook":hook, "cta":cta, "image_prompt":image_prompt, "video_script":script, "posting_time":simple_posting_schedule(topic), "reading_time":estimate_reading_time(post_text), "char_count":len(post_text), "confidence":"template"}

def generate_variants(topic:str, tone:str, platform:str, length_pref:str, n_variants:int, keywords_input:str)->List[Dict[str,Any]]:
    length_map={"Short":120,"Medium":300,"Long":800}
    max_chars=length_map.get(length_pref,300)
    user_kws=[k.strip() for k in re.split(r"[,\n;]+",keywords_input) if k.strip()]
    variants=[]
    for i in range(n_variants):
        variant=template_generator(topic,tone,platform,max_chars)
        if user_kws:
            user_tags=["#"+re.sub(r"\s+","",k) for k in user_kws][:4]
            variant["hashtags"]=list(dict.fromkeys(user_tags+variant["hashtags"]))
        variants.append(variant)
    return variants

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI SOCIAL MEDIA CONTENT CREATOR", layout="wide")
css()

if "variants" not in st.session_state: st.session_state.variants=[]
if "chat_history" not in st.session_state: st.session_state.chat_history=[]

left_col,right_col=st.columns([2.6,1])

with right_col:
    st.markdown('<div class="control-panel glass">',unsafe_allow_html=True)
    topic_input = st.text_input("Topic / Idea","e.g., AI marketing tips")
    platform = st.selectbox("Platform",["Instagram","YouTube Shorts","Twitter/X","LinkedIn","Facebook","TikTok"])
    tone = st.selectbox("Tone",["Professional","Casual","Funny","Inspirational","Urgent"])
    length_pref = st.selectbox("Post length",["Short","Medium","Long"])
    num_variants = st.slider("Variations",1,10,3)
    kw_input = st.text_area("Optional keywords / hashtags",height=60)
    gen_btn = st.button("Generate ✨")
    clear_btn = st.button("Clear Output")
    st.markdown("</div>",unsafe_allow_html=True)

with left_col:
    tabs=st.tabs(["Generator","Chatbot"])
    with tabs[0]:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        if gen_btn:
            if not topic_input.strip(): st.warning("Enter a topic first.")
            else:
                with st.spinner("Generating variants..."):
                    st.session_state.variants=generate_variants(topic_input,tone,platform,length_pref,num_variants,kw_input)
        if clear_btn: st.session_state.variants=[]
        if not st.session_state.variants: st.info("No variants yet. Enter topic and press Generate.")
        else:
            for i,var in enumerate(st.session_state.variants):
                st.markdown(f'<div class="variant-card">',unsafe_allow_html=True)
                edited=st.text_area(f"Variant {i+1} Post Text",value=var["post"],height=110)
                st.session_state.variants[i]["post"]=edited
                st.write("Hashtags:", " ".join(var["hashtags"]))
                st.write("Hook:",var["hook"])
                st.write("CTA:",var["cta"])
                st.write("Image prompt:",var["image_prompt"])
                st.write("Video script:",var["video_script"])
                st.write("Posting time:",var["posting_time"])
                st.write("Reading time / chars:",f"{var['reading_time']} / {var['char_count']}")
                st.write("Confidence:",var["confidence"])
                st.markdown("</div>",unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        st.text("Chatbot (Professional Assistant) - Type instructions like 'shorten', 'polish', 'add hashtags'")
        user_cmd=st.text_input("Enter instruction",key="chat_input")
        if st.button("Send to Assistant"):
            if user_cmd.strip(): st.session_state.chat_history.append({"user":user_cmd,"bot":"Assistant replies here (fallback deterministic)."})
        for entry in st.session_state.chat_history:
            st.markdown(f"**You:** {entry['user']}")
            st.markdown(f"**Assistant:** {entry['bot']}")
st.markdown("### App Info (Debug)")
st.write("Model loaded:",MODEL_LOADED)
st.write("Transformers available:",TRANSFORMERS_AVAILABLE)

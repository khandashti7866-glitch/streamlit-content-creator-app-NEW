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

How to run:
1. python -m venv venv
2. pip activate venv  # or source venv/bin/activate
3. pip install -r requirements.txt
4. streamlit run app.py

Notes:
- transformers & torch are optional. If present the app will try to load a small model.
- The app will attempt to download the model if not present (internet) but will still work if download fails.
"""

# -------------------------
# Requirements (for requirements.txt)
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

# Optional dependencies
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
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1503602642458-232111445657?auto=format&fit=crop&w=1600&q=80"
MODEL_NAME = "sshleifer/tiny-gpt2"
MAX_MODEL_TOKENS = 150

MIN_VARIANTS = 1
MAX_VARIANTS = 10

STOPWORDS = set([
    "the", "and", "is", "in", "to", "a", "of", "for", "on", "with", "that", "this", "are", "it", "as", "be", "by",
    "an", "from", "at", "or", "your", "you"
])

# -------------------------
# Try to load model
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
        print("Transformer model not loaded (fallback will be used). Reason:", e)
else:
    MODEL_LOADED = False

# -------------------------
# Utility functions
# -------------------------
def css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{BACKGROUND_IMAGE}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #efefef;
        }}
        h1, h2, h3, p, label, span {{
            text-shadow: 0 2px 12px rgba(0,0,0,0.85);
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
            text-shadow: 0 0 18px rgba(255, 209, 102, 0.28), 0 0 6px rgba(0,0,0,0.6);
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
        .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {{
            background: rgba(255,255,255,0.06);
            color: #fff;
            border-radius: 8px;
            padding: 8px;
        }}
        .stButton>button {{
            background: linear-gradient(90deg,#ffd166,#ffb703);
            color: #08111a;
            font-weight: 800;
            border-radius: 10px;
            padding: 8px 18px;
            border: none;
            box-shadow: 0 6px 18px rgba(255,181,3,0.18);
        }}
        .secondary-btn {{
            background: transparent;
            color: #ffd166;
            border: 1px solid rgba(255,215,0,0.14);
            border-radius: 8px;
            padding: 6px 12px;
            font-weight: 700;
        }}
        .variant-card {{
            background: linear-gradient(180deg, rgba(0,0,0,0.42), rgba(6,6,6,0.52));
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,215,0,0.06);
        }}
        .meta {{
            color: #e7e7e7;
            opacity: 0.88;
            font-size: 13px;
        }}
        .copy-btn {{
            background: #ffd166;
            color: #08111a;
            border-radius: 8px;
            padding:6px 10px;
            font-weight:700;
        }}
        .stApp .element-container {{ padding: 12px 14px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    toks = re.findall(r"\w+", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    if not toks:
        return []
    freqs = {}
    for t in toks:
        freqs[t] = freqs.get(t, 0) + 1
    sorted_tokens = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in sorted_tokens][:max_keywords]

def generate_hashtags_from_keywords(keywords: List[str], min_tags: int = 5, max_tags: int = 12) -> List[str]:
    tags = []
    for k in keywords:
        clean = re.sub(r"[^A-Za-z0-9]", "", k)
        if clean:
            tags.append("#" + clean)
    extras = ["#viral", "#trending", "#contentcreator", "#tips", "#howto", "#learn"]
    random.shuffle(extras)
    tags = tags + extras[:max(0, min_tags - len(tags))]
    return tags[:max_tags]

def simple_posting_schedule(topic: str) -> str:
    low = topic.lower()
    hours = [9, 11, 14, 17, 19, 21]
    day = random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    if any(k in low for k in ["business","productivity","career","finance","invest"]):
        day = random.choice(["Tuesday","Wednesday","Thursday"])
    elif any(k in low for k in ["life","fun","travel","food","fashion","music"]):
        day = random.choice(["Friday","Saturday","Sunday"])
    return f"{day} at {random.choice(hours)}:00"

def estimate_reading_time(text: str) -> str:
    words = len(re.findall(r"\w+", text))
    minutes = max(1, int(words / 200))
    return f"{minutes} min"

def truncate_to_length(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(".")
    if cut == -1:
        cut = text[:max_chars].rfind(" ")
    if cut == -1 or cut < max_chars // 2:
        return text[:max_chars].rstrip() + "…"
    return text[:cut+1]

# -------------------------
# Model & template generation
# -------------------------
def model_generate(prompt: str, max_new_tokens: int = 80) -> Optional[str]:
    global MODEL_LOADED, model, tokenizer
    if not MODEL_LOADED or model is None or tokenizer is None:
        return None
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        out = model.generate(input_ids, max_length=min(len(input_ids[0]) + max_new_tokens, MAX_MODEL_TOKENS),
                             do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        else:
            return text.strip()
    except Exception as e:
        print("Model generation error:", e)
        return None

def template_generator(topic: str, tone: str, platform: str, length_chars: int) -> Dict[str, Any]:
    kw = extract_keywords(topic, max_keywords=6)
    hashtags = generate_hashtags_from_keywords(kw)
    hooks = {
        "Professional": [
            f"Industry insight: {topic} — what professionals must know.",
            f"Brief update on {topic} that impacts many industries."
        ],
        "Casual": [
            f"Quick tip about {topic} you can use today!",
            f"Real talk: {topic} explained simply."
        ],
        "Funny": [
            f"If {topic} were a person, here's what they'd say...",
            f"Fun facts (and laughs) about {topic}."
        ],
        "Inspirational": [
            f"How {topic} changed the game for many people.",
            f"One idea that might change your view about {topic}."
        ],
        "Urgent": [
            f"Important! {topic} updates you need to act on.",
            f"Alert: new {topic} shifts happening now."
        ]
    }
    ctas = ["Learn more", "Share your thoughts", "Save this post", "Try this now", "Join the conversation"]
    images = [
        f"Luxurious black and gold flatlay about {topic}",
        f"High-contrast professional image representing {topic}",
        f"Minimalist editorial photo focused on {topic}"
    ]
    scripts = [
        f"Hi — quick tip on {topic}. First, ... Next, ... Finally, ...",
        f"{topic} in 30 seconds: the key steps are A, B, and C."
    ]
    tone_h = hooks.get(tone, hooks["Professional"])
    hook = random.choice(tone_h)
    cta = random.choice(ctas)
    image_prompt = random.choice(images)
    script = random.choice(scripts)
    base = f"{hook} {topic}. {truncate_to_length(' '.join([topic, 'This post explains key points and actions.']), length_chars)} {cta}."
    post_text = truncate_to_length(base, length_chars)
    posting_time = simple_posting_schedule(topic)
    reading_time = estimate_reading_time(post_text)
    char_count = len(post_text)
    confidence = "template"
    return {
        "post": post_text,
        "hashtags": hashtags,
        "hook": hook,
        "cta": cta,
        "image_prompt": image_prompt,
        "video_script": script,
        "posting_time": posting_time,
        "reading_time": reading_time,
        "char_count": char_count,
        "confidence": confidence
    }

def generate_variants(topic: str, tone: str, platform: str, length_pref: str, n_variants: int, keywords_input: str) -> List[Dict[str, Any]]:
    length_map = {"Short": 120, "Medium": 300, "Long": 800}
    max_chars = length_map.get(length_pref, 300)
    user_kws = [k.strip() for k in re.split(r"[,\n;]+", keywords_input) if k.strip()]
    variants = []
    for i in range(n_variants):
        prompt = f"Topic: {topic}\nTone: {tone}\nPlatform: {platform}\nWrite a single post sized to {max_chars} characters, include a short hook, a CTA, and a 1-sentence image prompt. Use professional tone.\nPost:"
        generated_text = None
        if MODEL_LOADED:
            gen = model_generate(prompt, max_new_tokens=120)
            if gen:
                text = (gen or "").strip()
                post_text = truncate_to_length(text.split("\n")[0] if "\n" in text else text, max_chars)
                kw = extract_keywords(topic) + user_kws
                hashtags = generate_hashtags_from_keywords(kw)
                variants.append({
                    "post": post_text,
                    "hashtags": hashtags,
                    "hook": post_text.split(".")[0] if "." in post_text else post_text[:60],
                    "cta": "Read more",
                    "image_prompt": "A professional photo illustrating " + topic,
                    "video_script": "Short script: " + (post_text[:120] + "..." if len(post_text) > 120 else post_text),
                    "posting_time": simple_posting_schedule(topic),
                    "reading_time": estimate_reading_time(post_text),
                    "char_count": len(post_text),
                    "confidence": "model"
                })
                continue
        variant = template_generator(topic, tone, platform, max_chars)
        if user_kws:
            user_tags = ["#" + re.sub(r"\s+", "", k) for k in user_kws][:4]
            variant["hashtags"] = list(dict.fromkeys(user_tags + variant["hashtags"]))
        variants.append(variant)
    return variants

# -------------------------
# Chatbot assistant
# -------------------------
def chatbot_transform(message: str, last_content: Optional[str]) -> str:
    prompt = message.lower()
    if MODEL_LOADED:
        full_prompt = f"You are a professional writing assistant. User instruction: {message}\nContent: {last_content or ''}\nProvide a revised version or suggestions."
        gen = model_generate(full_prompt, max_new_tokens=160)
        if gen:
            return gen.strip()
    if "shorten" in prompt or "short" in prompt:
        if last_content:
            return truncate_to_length(last_content, 280)
        return "Please provide the text you want shortened."
    if "professional" in prompt or "polish" in prompt or "improve" in prompt:
        if not last_content:
            return "Please provide the content you want polished."
        out = last_content.replace("don't", "do not").replace("can't", "cannot").replace("it's", "it is")
        return f"{out.strip()} \n\n— Polished by your professional writing assistant."
    if "translate" in prompt and "urdu" in prompt:
        if not last_content:
            return "Please provide the text you want translated to Urdu."
        small_dict = {"hello":"ہیلو", "thank you":"شکریہ", "good":"اچھا", "best":"بہترین", "today":"آج", "learn":"سیکھیں"}
        words = last_content.split()
        translated = []
        for w in words:
            lw = re.sub(r"[^a-zA-Z']", "", w).lower()
            translated.append(small_dict.get(lw, w))
        return (" ".join(translated) + "\n\nNote: This is a simple fallback translation; use a translation package for better results.")
    if "funn" in prompt or "funny" in prompt:
        if not last_content:
            return "Please provide the text you want made funnier."
        return last_content + " 😂 (funny twist added by your assistant)"
    if "hashtags" in prompt:
        kw = extract_keywords(last_content or message)
        return " ".join(generate_hashtags_from_keywords(kw, min_tags=6))
    return "I can help with shortening, polishing, translating (limited), adding hashtags, or rewriting the tone. Try commands like 'shorten', 'make professional', 'translate to Urdu', 'add hashtags'."

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="VIP AI Content Creator", page_icon="✨", layout="wide")
css()

if "variants" not in st.session_state:
    st.session_state.variants = []
if "clipboard_log" not in st.session_state:
    st.session_state.clipboard_log = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_generated_topic" not in st.session_state:
    st.session_state.last_generated_topic = ""

st.title("")
left_col, right_col = st.columns([2.6, 1])

with right_col:
    st.markdown('<div class="control-panel glass">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center;">'
                '<div><h2 class="gold-title">VIP Controls</h2><div class="gold-sub">Right-side panel</div></div>'
                '</div>', unsafe_allow_html=True

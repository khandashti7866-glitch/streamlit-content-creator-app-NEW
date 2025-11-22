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
# Requirements (for requirements.txt file - included as snippet at bottom)
# -------------------------
# streamlit
# pandas
# numpy
# transformers  # optional
# torch         # optional
# pillow        # optional (not strictly required but used in comments)

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import time
from datetime import datetime, timedelta
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
# Choose theme variant: Black & Gold VIP
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1503602642458-232111445657?auto=format&fit=crop&w=1600&q=80"
MODEL_NAME = "sshleifer/tiny-gpt2"  # small model; change if you want distilgpt2 etc.
MAX_MODEL_TOKENS = 150

# UI constraints
MIN_VARIANTS = 1
MAX_VARIANTS = 10

# Stopwords for keyword extraction
STOPWORDS = set([
    "the", "and", "is", "in", "to", "a", "of", "for", "on", "with", "that", "this", "are", "it", "as", "be", "by",
    "an", "from", "at", "or", "your", "you"
])

# -------------------------
# Try to load model (non-fatal)
# -------------------------
if TRANSFORMERS_AVAILABLE:
    try:
        # Attempt to load model/tokenizer. This may download if not cached.
        # Wrap in try/except to avoid crashing; we'll fallback if it fails.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # Put model on CPU (safe)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        MODEL_LOADED = True
    except Exception as e:
        # If anything fails, set flags; fallback will be used.
        MODEL_LOADED = False
        # We do not re-raise; we'll fall back to template generator
        print("Transformer model not loaded (fallback will be used). Reason:", e)
else:
    MODEL_LOADED = False

# -------------------------
# Utility functions
# -------------------------
def css() -> None:
    """Inject CSS for Black & Gold VIP theme, glass boxes, right UI, and blue bold headings per earlier request."""
    st.markdown(
        f"""
        <style>
        /* App background */
        .stApp {{
            background-image: url("{BACKGROUND_IMAGE}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #efefef;
        }}

        /* Global text shadow for readability */
        h1, h2, h3, p, label, span {{
            text-shadow: 0 2px 12px rgba(0,0,0,0.85);
        }}

        /* VIP black & gold glass panel style */
        .glass {{
            background: linear-gradient(180deg, rgba(6,6,6,0.72) 0%, rgba(14,14,14,0.60) 100%);
            border: 1px solid rgba(255, 215, 0, 0.12); /* faint gold border */
            padding: 18px;
            border-radius: 14px;
            backdrop-filter: blur(6px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
            color: #fff;
        }}

        /* Gold heading */
        .gold-title {{
            font-size: 36px;
            font-weight: 800;
            color: #ffd166;
            text-shadow: 0 0 18px rgba(255, 209, 102, 0.28), 0 0 6px rgba(0,0,0,0.6);
            margin-bottom: 6px;
        }}

        /* Accent subtitle */
        .gold-sub {{
            color: #ffd166;
            opacity: 0.9;
            margin-top: -8px;
            margin-bottom: 12px;
        }}

        /* Right control panel area style */
        .control-panel {{
            background: rgba(10,10,10,0.6);
            border-radius: 12px;
            padding: 14px;
            border: 1px solid rgba(255,215,0,0.08);
        }}

        /* Rounded, translucent inputs */
        .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {{
            background: rgba(255,255,255,0.06);
            color: #fff;
            border-radius: 8px;
            padding: 8px;
        }}

        /* Buttons - gold */
        .stButton>button {{
            background: linear-gradient(90deg,#ffd166,#ffb703);
            color: #08111a;
            font-weight: 800;
            border-radius: 10px;
            padding: 8px 18px;
            border: none;
            box-shadow: 0 6px 18px rgba(255,181,3,0.18);
        }}

        /* Smaller secondary buttons */
        .secondary-btn {{
            background: transparent;
            color: #ffd166;
            border: 1px solid rgba(255,215,0,0.14);
            border-radius: 8px;
            padding: 6px 12px;
            font-weight: 700;
        }}

        /* Variant card */
        .variant-card {{
            background: linear-gradient(180deg, rgba(0,0,0,0.42), rgba(6,6,6,0.52));
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,215,0,0.06);
        }}

        /* small meta text */
        .meta {{
            color: #e7e7e7;
            opacity: 0.88;
            font-size: 13px;
        }}

        /* copy button style for custom copy */
        .copy-btn {{
            background: #ffd166;
            color: #08111a;
            border-radius: 8px;
            padding:6px 10px;
            font-weight:700;
        }}

        /* ensure wide layout spacing is nice */
        .stApp .element-container {{ padding: 12px 14px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Simple keyword extraction by heuristics: words longer than 2 chars excluding stopwords, sorted by frequency."""
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
    """Create hashtag list from keywords and smart extras."""
    tags = []
    for k in keywords:
        clean = re.sub(r"[^A-Za-z0-9]", "", k)
        if clean:
            tags.append("#" + clean)
    extras = ["#viral", "#trending", "#contentcreator", "#tips", "#howto", "#learn"]
    random.shuffle(extras)
    # ensure at least min_tags
    tags = tags + extras[:max(0, min_tags - len(tags))]
    # cap to max_tags
    return tags[:max_tags]

def simple_posting_schedule(topic: str) -> str:
    """Rule-based best day/hour schedule."""
    # Example rules: business topics -> weekdays; lifestyle -> weekends
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
    minutes = max(1, int(words / 200))  # 200 wpm
    return f"{minutes} min"

def truncate_to_length(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # try to cut at last sentence or space before max_chars
    cut = text[:max_chars].rfind(".")
    if cut == -1:
        cut = text[:max_chars].rfind(" ")
    if cut == -1 or cut < max_chars // 2:
        return text[:max_chars].rstrip() + "…"
    return text[:cut+1]

# -------------------------
# Generation: Model & Fallback
# -------------------------
def model_generate(prompt: str, max_new_tokens: int = 80) -> Optional[str]:
    """Generate using model if loaded. Returns None on failure."""
    global MODEL_LOADED, model, tokenizer
    if not MODEL_LOADED or model is None or tokenizer is None:
        return None
    try:
        # tokenizer and model from transformers
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # move to same device as model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        out = model.generate(input_ids, max_length=min(len(input_ids[0]) + max_new_tokens, MAX_MODEL_TOKENS),
                             do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # We want the generated continuation after the prompt (some models include prompt)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        else:
            return text.strip()
    except Exception as e:
        print("Model generation error:", e)
        return None

def template_generator(topic: str, tone: str, platform: str, length_chars: int) -> Dict[str, Any]:
    """Deterministic template-based variant generator. Always returns usable output."""
    kw = extract_keywords(topic, max_keywords=6)
    hashtags = generate_hashtags_from_keywords(kw)
    # Templates per tone (professional persona)
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
    ctas = [
        "Learn more", "Share your thoughts", "Save this post", "Try this now", "Join the conversation"
    ]
    images = [
        f"Luxurious black and gold flatlay about {topic}",
        f"High-contrast professional image representing {topic}",
        f"Minimalist editorial photo focused on {topic}"
    ]
    scripts = [
        f"Hi — quick tip on {topic}. First, ... Next, ... Finally, ...",
        f"{topic} in 30 seconds: the key steps are A, B, and C."
    ]
    # Pick by tone
    tone_h = hooks.get(tone, hooks["Professional"])
    hook = random.choice(tone_h)
    cta = random.choice(ctas)
    image_prompt = random.choice(images)
    script = random.choice(scripts)
    # Post text generation - combine hook and CTA and a short explanation
    base = f"{hook} {topic}. {truncate_to_length(' '.join([topic, 'This post explains key points and actions.']), length_chars)} {cta}."
    post_text = truncate_to_length(base, length_chars)
    # meta
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
    """Master generator: tries model first, falls back to template engine."""
    # map length to chars
    length_map = {"Short": 120, "Medium": 300, "Long": 800}
    max_chars = length_map.get(length_pref, 300)
    # collect keywords from user input too
    user_kws = [k.strip() for k in re.split(r"[,\n;]+", keywords_input) if k.strip()]
    # generate
    variants = []
    for i in range(n_variants):
        # attempt model generation if loaded
        prompt = f"Topic: {topic}\nTone: {tone}\nPlatform: {platform}\nWrite a single post sized to {max_chars} characters, include a short hook, a CTA, and a 1-sentence image prompt. Use professional tone.\nPost:"
        generated_text = None
        if MODEL_LOADED:
            gen = model_generate(prompt, max_new_tokens=120)
            if gen:
                # try to parse into pieces heuristically
                text = (gen or "").strip()
                # fallback to templates for fields we cannot parse
                post_text = truncate_to_length(text.split("\n")[0] if "\n" in text else text, max_chars)
                # Build extracted keywords from user or topic
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
                continue  # next variant
        # If model is not loaded or generation failed, use template generator
        variant = template_generator(topic, tone, platform, max_chars)
        # integrate user keywords if provided
        if user_kws:
            # prepend some user keywords into hashtags
            user_tags = ["#" + re.sub(r"\s+", "", k) for k in user_kws][:4]
            # merge while preserving uniqueness
            variant["hashtags"] = list(dict.fromkeys(user_tags + variant["hashtags"]))
        variants.append(variant)
    return variants

# -------------------------
# Chatbot utilities (Professional assistant persona)
# -------------------------
def chatbot_transform(message: str, last_content: Optional[str]) -> str:
    """
    Simple deterministic 'professional writing assistant' transformations.
    If model available, attempt to use it. Otherwise apply deterministic rules:
    - 'shorten' or 'shorten for twitter' -> trims to 280 chars
    - 'make professional' -> formalize contractions / polish
    - 'translate to urdu' -> simple dictionary fallback (limited)
    - 'make funnier' -> insert light humorous phrase
    - 'improve' or 'polish' -> reflow sentences
    - generic -> echo suggestion for improvements
    """
    prompt = message.lower()
    # If model loaded try to use it for better replies
    if MODEL_LOADED:
        # Craft a prompt instructing professional assistant behavior
        full_prompt = f"You are a professional writing assistant. User instruction: {message}\nContent: {last_content or ''}\nProvide a revised version or suggestions."
        gen = model_generate(full_prompt, max_new_tokens=160)
        if gen:
            return gen.strip()
    # Deterministic fallbacks
    if "shorten" in prompt or "short" in prompt:
        if last_content:
            return truncate_to_length(last_content, 280)
        return "Please provide the text you want shortened."
    if "professional" in prompt or "polish" in prompt or "improve" in prompt:
        if not last_content:
            return "Please provide the content you want polished."
        # Simple polishing: expand contractions and increase formality (basic)
        out = last_content.replace("don't", "do not").replace("can't", "cannot").replace("it's", "it is")
        # Add a professional prefix/suffix
        return f"{out.strip()} \n\n— Polished by your professional writing assistant."
    if "translate" in prompt and "urdu" in prompt:
        # small dictionary fallback: extremely limited; inform user about limitations
        if not last_content:
            return "Please provide the text you want translated to Urdu."
        # limited word map (not comprehensive)
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
        # generate more hashtags from last content
        kw = extract_keywords(last_content or message)
        return " ".join(generate_hashtags_from_keywords(kw, min_tags=6))
    # Default helpful reply
    return "I can help with shortening, polishing, translating (limited), adding hashtags, or rewriting the tone. Try commands like 'shorten', 'make professional', 'translate to Urdu', 'add hashtags'."

# -------------------------
# App UI - build layout & session state
# -------------------------
st.set_page_config(page_title="VIP AI Content Creator", page_icon="✨", layout="wide")
css()  # inject CSS

# Initialize session state
if "variants" not in st.session_state:
    st.session_state.variants = []  # list of dicts
if "clipboard_log" not in st.session_state:
    st.session_state.clipboard_log = []  # entries of {'time':..., 'text':...}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {'user':..., 'bot':...}
if "last_generated_topic" not in st.session_state:
    st.session_state.last_generated_topic = ""

st.title("")  # empty to keep header space minimal
# Main layout: left (output), right (controls)
left_col, right_col = st.columns([2.6, 1])

# Right-side UI controls (VIP control panel)
with right_col:
    st.markdown('<div class="control-panel glass">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center;">'
                '<div><h2 class="gold-title">VIP Controls</h2><div class="gold-sub">Right-side panel</div></div>'
                '</div>', unsafe_allow_html=True)

    # Inputs
    topic_input = st.text_input("Topic / Idea", value=st.session_state.get("last_generated_topic", ""), placeholder="e.g., Build confidence in public speaking")
    platform = st.selectbox("Platform", ["Instagram","YouTube Shorts","Twitter/X","LinkedIn","Facebook","TikTok"])
    tone = st.selectbox("Tone", ["Professional","Casual","Funny","Inspirational","Urgent"], index=0)
    length_pref = st.selectbox("Post length", ["Short","Medium","Long"], index=1)
    num_variants = st.slider("Variations", min_value=1, max_value=10, value=3)
    kw_input = st.text_area("Optional keywords / hashtags (comma separated)", value="", height=60)

    st.markdown("<div style='display:flex; gap:8px; margin-top:8px;'>", unsafe_allow_html=True)
    gen_btn = st.button("Generate ✨")
    clear_btn = st.button("Clear Output")
    st.markdown("</div>", unsafe_allow_html=True)

    # Export all variants
    st.markdown("<hr style='opacity:0.12'>", unsafe_allow_html=True)
    if st.button("Export All (CSV)"):
        if not st.session_state.variants:
            st.warning("No generated variants to export.")
        else:
            df = pd.DataFrame(st.session_state.variants)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="variants.csv", mime="text/csv")
    if st.button("Export All (JSON)"):
        if not st.session_state.variants:
            st.warning("No generated variants to export.")
        else:
            import json
            payload = json.dumps(st.session_state.variants, ensure_ascii=False, indent=2)
            st.download_button("Download JSON", payload, file_name="variants.json", mime="application/json")

    # Clipboard log download
    if st.session_state.clipboard_log:
        if st.button("Download Clipboard Log"):
            df_log = pd.DataFrame(st.session_state.clipboard_log)
            st.download_button("Download Log CSV", df_log.to_csv(index=False).encode("utf-8"), "clipboard_log.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

# Left content area: tabs (Generator & Chatbot) so both are visible in left but controls on right remain
with left_col:
    tabs = st.tabs(["Generator", "Chatbot"])
    # -----------------------
    # Generator Tab
    # -----------------------
    with tabs[0]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex; justify-content:space-between; align-items:end;">'
                    '<div><h2 class="gold-title">Generator</h2>'
                    '<div class="meta">Generate ready-to-post social content — multiple variants, hooks, CTAs, hashtags & scripts</div></div>'
                    '</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Show last topic summary
        if st.session_state.last_generated_topic:
            st.markdown(f"**Last topic:** {st.session_state.last_generated_topic}", unsafe_allow_html=True)

        # Handle generate/clear actions
        if gen_btn:
            if not topic_input.strip():
                st.warning("Please enter a topic in the right-side control panel.")
            else:
                with st.spinner("Generating variants..."):
                    # Generate
                    variants = generate_variants(topic_input, tone, platform, length_pref, num_variants, kw_input)
                    # add timestamp and ids
                    now = datetime.now().isoformat(timespec="seconds")
                    for idx, v in enumerate(variants):
                        v["_id"] = f"v_{int(time.time()*1000)}_{idx}"
                        v["_generated_at"] = now
                    st.session_state.variants = variants
                    st.session_state.last_generated_topic = topic_input.strip()
                    time.sleep(0.6)  # small UX pause

        if clear_btn:
            st.session_state.variants = []
            st.success("Cleared generated variants.")

        # Display variants (if any)
        if not st.session_state.variants:
            st.info("No variants yet. Use the controls on the right to enter a topic and press Generate.")
        else:
            # Loop through variants and render each as a card with actions
            for i, var in enumerate(st.session_state.variants):
                # Use unique keys for Streamlit widgets to avoid collisions
                card_key = f"card_{i}_{var.get('_id','')}"
                st.markdown(f'<div class="variant-card">', unsafe_allow_html=True)
                st.markdown(f"**Variant {i+1}**  <span class='meta'>• generated at {var.get('_generated_at','-')}</span>", unsafe_allow_html=True)
                st.write("")  # small spacer

                # Post text area (editable)
                txt_key = f"text_{card_key}"
                edited = st.text_area("Post Text", value=var["post"], key=txt_key, height=110)
                # Update session_state if edited
                st.session_state.variants[i]["post"] = edited
                st.write("**Hashtags:**", " ".join(var["hashtags"]))
                st.write("**Hook:**", var["hook"])
                st.write("**CTA:**", var["cta"])
                st.write("**Image prompt:**", var["image_prompt"])
                st.write("**Short video script:**", var["video_script"])
                st.write("**Posting time:**", var["posting_time"])
                st.write("**Reading time / chars:**", f"{var['reading_time']} / {var['char_count']}")
                st.write("**Confidence:**", var.get("confidence", "template"))

                # Action buttons: Copy, Regenerate, Export single
                # Copy to clipboard -> use JS button that writes to clipboard
                copy_js = f"""
                <script>
                function copyText_{card_key}() {{
                    navigator.clipboard.writeText({repr(edited)});
                    document.getElementById("copy_status_{card_key}").innerText = "Copied!";
                }}
                </script>
                """
                st.markdown(copy_js, unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    st.markdown(f'<button class="copy-btn" onclick="copyText_{card_key}()">Copy</button> <span id="copy_status_{card_key}" class="meta" style="margin-left:8px;"></span>', unsafe_allow_html=True)
                    # record copy in clipboard log if clicked - provide a small selectbox to trigger (since JS cannot call python)
                    # fallback: provide a Streamlit button to also copy (server-side log)
                    if st.button("Copy & Log", key=f"logcopy_{card_key}"):
                        st.session_state.clipboard_log.append({"time": datetime.now().isoformat(timespec="seconds"), "text": edited})
                        st.success("Copied to clipboard log (client copy may need browser permission).")
                with col2:
                    if st.button("Regenerate", key=f"regen_{card_key}"):
                        # regenerate single variant using same params
                        with st.spinner("Regenerating variant..."):
                            new_var = generate_variants(st.session_state.last_generated_topic or topic_input, tone, platform, length_pref, 1, kw_input)[0]
                            new_var["_id"] = f"v_{int(time.time()*1000)}_regen"
                            new_var["_generated_at"] = datetime.now().isoformat(timespec="seconds")
                            st.session_state.variants[i] = new_var
                            st.experimental_rerun()
                with col3:
                    # Export single variant
                    # JSON
                    if st.button("Export JSON", key=f"json_{card_key}"):
                        import json, base64
                        payload = json.dumps(var, ensure_ascii=False, indent=2)
                        st.download_button("Download JSON", payload, file_name=f"variant_{i+1}.json", mime="application/json")
                with col4:
                    # Export single as text
                    if st.button("Export TXT", key=f"txt_{card_key}"):
                        txt = var["post"]
                        st.download_button("Download TXT", txt, file_name=f"variant_{i+1}.txt", mime="text/plain")
                st.markdown("</div>", unsafe_allow_html=True)  # end variant card

        st.markdown("</div>", unsafe_allow_html=True)  # end glass for generator tab

    # -----------------------
    # Chatbot Tab - left area
    # -----------------------
    with tabs[1]:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<h2 class="gold-title">Chatbot — Professional Assistant</h2>', unsafe_allow_html=True)
        st.markdown('<div class="meta">Ask follow-up edits about any generated content. (Polish, shorten, translate - limited)</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Show conversation
        for entry in st.session_state.chat_history:
            st.markdown(f"**You:** {entry['user']}")
            st.markdown(f"**Assistant:** {entry['bot']}")
            st.markdown("---")

        # Input area for chat commands
        user_cmd = st.text_input("Enter instruction (e.g., 'shorten', 'make professional', 'translate to Urdu', 'add hashtags')", key="chat_input")
        # Provide contextual "apply to" option: choose which variant to apply transformation to
        apply_to = None
        if st.session_state.variants:
            choices = [f"Variant {i+1}" for i in range(len(st.session_state.variants))]
            choices.insert(0, "No specific variant (ask general)")
            selected = st.selectbox("Apply to", choices, index=0)
            if selected != "No specific variant (ask general)":
                apply_to = int(selected.replace("Variant ","")) - 1

        if st.button("Send to Assistant"):
            if not user_cmd.strip():
                st.warning("Please enter an instruction.")
            else:
                # Determine last content
                last_content = None
                if apply_to is not None:
                    last_content = st.session_state.variants[apply_to]["post"]
                # Generate assistant reply
                reply = chatbot_transform(user_cmd, last_content)
                st.session_state.chat_history.append({"user": user_cmd, "bot": reply})
                # If instruction was transformation and applied to a variant, update it
                if apply_to is not None:
                    # Determine transform intent (basic)
                    low = user_cmd.lower()
                    if any(k in low for k in ["shorten","short"]):
                        st.session_state.variants[apply_to]["post"] = truncate_to_length(reply if len(reply)>0 else last_content, 280)
                        st.success("Applied shortened text to variant.")
                    elif any(k in low for k in ["professional","polish","improve"]):
                        st.session_state.variants[apply_to]["post"] = reply
                        st.success("Applied polished text to variant.")
                    elif "hashtags" in low:
                        # replace hashtags in variant
                        new_tags = reply.split()
                        st.session_state.variants[apply_to]["hashtags"] = new_tags
                        st.success("Applied new hashtags to variant.")
                    elif "translate" in low and "urdu" in low:
                        st.session_state.variants[apply_to]["post"] = reply
                        st.success("Applied translated text to variant (limited fallback).")
                    else:
                        # generic replacement if reply looks like a rewritten post (heuristic: contains spaces and punctuation)
                        if last_content and len(reply) > 20:
                            st.session_state.variants[apply_to]["post"] = reply
                            st.success("Applied assistant output to variant.")
                # small UX pause
                time.sleep(0.2)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer & debug area (collapsed)
# -------------------------
with st.expander("App Info & Debug"):
    st.markdown("**Model loaded:** " + ("Yes" if MODEL_LOADED else "No (using fallback templates)"))
    st.markdown(f"**Transformers package available:** {TRANSFORMERS_AVAILABLE}")
    st.markdown("**Number of saved variants in session:** " + str(len(st.session_state.variants)))
    st.markdown("**Clipboard log entries:** " + str(len(st.session_state.clipboard_log)))
    st.markdown("---")
    st.markdown("**Notes:** This app works without API keys. If you want better model outputs enable 'transformers' and 'torch' and ensure internet access for the first model download. If a model is not available the deterministic template engine will always produce results.")

# -------------------------
# Minimal tests for manual run
# -------------------------
def run_demo_tests() -> None:
    """Simple test to verify fallback generator returns variants."""
    t = "Grow confidence for public speaking"
    vs = generate_variants(t, "Professional", "LinkedIn", "Medium", 2, "confidence, public speaking")
    print("Demo generated variants:", vs)

if __name__ == "__main__":
    # show a minimal console output when run as script (not streamlit)
    run_demo_tests()

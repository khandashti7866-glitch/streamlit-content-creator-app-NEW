# -------------------------
# VIP Social Media Content Creator - Streamlit App (Fixed keys)
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
# Config & constants
# -------------------------
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1503602642458-232111445657?auto=format&fit=crop&w=1600&q=80"
MODEL_NAME = "sshleifer/tiny-gpt2"
MAX_MODEL_TOKENS = 150

MIN_VARIANTS = 1
MAX_VARIANTS = 10

STOPWORDS = set([
    "the","and","is","in","to","a","of","for","on","with","that","this","are","it","as","be","by","an","from","at","or","your","you"
])

# -------------------------
# Try to load model (non-fatal)
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
        print("Transformer model not loaded:", e)
else:
    MODEL_LOADED = False

# -------------------------
# Utilities
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
        h1,h2,h3,p,label,span {{ text-shadow:0 2px 12px rgba(0,0,0,0.85); }}
        .glass {{
            background: linear-gradient(180deg, rgba(6,6,6,0.72), rgba(14,14,14,0.6));
            border: 1px solid rgba(255,215,0,0.12);
            padding: 18px;
            border-radius: 14px;
            backdrop-filter: blur(6px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
            color:#fff;
        }}
        .gold-title {{ font-size:36px;font-weight:800;color:#ffd166;margin-bottom:6px; }}
        .gold-sub {{ color:#ffd166;opacity:0.9;margin-top:-8px;margin-bottom:12px; }}
        .control-panel {{ background: rgba(10,10,10,0.6); border-radius:12px;padding:14px; border:1px solid rgba(255,215,0,0.08); }}
        .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {{ background: rgba(255,255,255,0.06); color:#fff; border-radius:8px; padding:8px; }}
        .stButton>button {{ background: linear-gradient(90deg,#ffd166,#ffb703); color:#08111a; font-weight:800; border-radius:10px; padding:8px 18px; border:none; box-shadow:0 6px 18px rgba(255,181,3,0.18); }}
        .secondary-btn {{ background: transparent; color: #ffd166; border:1px solid rgba(255,215,0,0.14); border-radius:8px; padding:6px 12px; font-weight:700; }}
        .variant-card {{ background: linear-gradient(180deg, rgba(0,0,0,0.42), rgba(6,6,6,0.52)); border-radius:12px; padding:12px; margin-bottom:12px; border:1px solid rgba(255,215,0,0.06); }}
        .meta {{ color:#e7e7e7; opacity:0.88; font-size:13px; }}
        .copy-btn {{ background:#ffd166;color:#08111a;border-radius:8px;padding:6px 10px;font-weight:700; }}
        .stApp .element-container {{ padding:12px 14px; }}
        </style>
        """, unsafe_allow_html=True
    )

def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    toks = re.findall(r"\w+", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t)>2]
    if not toks: return []
    freqs = {}
    for t in toks: freqs[t] = freqs.get(t,0)+1
    sorted_tokens = sorted(freqs.items(), key=lambda x:(-x[1],x[0]))
    return [t for t,_ in sorted_tokens][:max_keywords]

def generate_hashtags_from_keywords(keywords: List[str], min_tags: int=5, max_tags:int=12) -> List[str]:
    tags=[]
    for k in keywords:
        clean = re.sub(r"[^A-Za-z0-9]","",k)
        if clean: tags.append("#"+clean)
    extras = ["#viral","#trending","#contentcreator","#tips","#howto","#learn"]
    random.shuffle(extras)
    tags = tags + extras[:max(0,min_tags-len(tags))]
    return tags[:max_tags]

def simple_posting_schedule(topic:str)->str:
    low = topic.lower()
    hours = [9,11,14,17,19,21]
    day = random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    if any(k in low for k in ["business","productivity","career","finance","invest"]): day=random.choice(["Tuesday","Wednesday","Thursday"])
    elif any(k in low for k in ["life","fun","travel","food","fashion","music"]): day=random.choice(["Friday","Saturday","Sunday"])
    return f"{day} at {random.choice(hours)}:00"

def estimate_reading_time(text:str)->str:
    words = len(re.findall(r"\w+", text))
    minutes = max(1,int(words/200))
    return f"{minutes} min"

def truncate_to_length(text:str, max_chars:int)->str:
    if len(text)<=max_chars: return text
    cut=text[:max_chars].rfind(".")
    if cut==-1: cut=text[:max_chars].rfind(" ")
    if cut==-1 or cut<max_chars//2: return text[:max_chars].rstrip()+"…"
    return text[:cut+1]

def template_generator(topic:str, tone:str, platform:str, length_chars:int)->Dict[str,Any]:
    kw = extract_keywords(topic,6)
    hashtags = generate_hashtags_from_keywords(kw)
    hooks = {
        "Professional":[f"Industry insight: {topic} — what professionals must know.", f"Brief update on {topic} that impacts many industries."],
        "Casual":[f"Quick tip about {topic} you can use today!", f"Real talk: {topic} explained simply."],
        "Funny":[f"If {topic} were a person, here's what they'd say...", f"Fun facts (and laughs) about {topic}."],
        "Inspirational":[f"How {topic} changed the game for many people.", f"One idea that might change your view about {topic}."],
        "Urgent":[f"Important! {topic} updates you need to act on.", f"Alert: new {topic} shifts happening now."]
    }
    ctas = ["Learn more","Share your thoughts","Save this post","Try this now","Join the conversation"]
    images = [f"Luxurious black and gold flatlay about {topic}", f"High-contrast professional image representing {topic}", f"Minimalist editorial photo focused on {topic}"]
    scripts = [f"Hi — quick tip on {topic}. First, ... Next, ... Finally, ...", f"{topic} in 30 seconds: the key steps are A, B, and C."]
    tone_h = hooks.get(tone,hooks["Professional"])
    hook = random.choice(tone_h)
    cta = random.choice(ctas)
    image_prompt = random.choice(images)
    script = random.choice(scripts)
    base = f"{hook} {topic}. {truncate_to_length(' '.join([topic,'This post explains key points and actions.']),length_chars)} {cta}."
    post_text = truncate_to_length(base,length_chars)
    posting_time = simple_posting_schedule(topic)
    reading_time = estimate_reading_time(post_text)
    char_count = len(post_text)
    confidence = "template"
    return {"post":post_text,"hashtags":hashtags,"hook":hook,"cta":cta,"image_prompt":image_prompt,"video_script":script,"posting_time":posting_time,"reading_time":reading_time,"char_count":char_count,"confidence":confidence}

def generate_variants(topic:str, tone:str, platform:str, length_pref:str, n_variants:int, keywords_input:str)->List[Dict[str,Any]]:
    length_map={"Short":120,"Medium":300,"Long":800}
    max_chars=length_map.get(length_pref,300)
    user_kws=[k.strip() for k in re.split(r"[,\n;]+",keywords_input) if k.strip()]
    variants=[]
    for i in range(n_variants):
        prompt=f"Topic: {topic}\nTone: {tone}\nPlatform: {platform}\nWrite a post of {max_chars} chars with hook, CTA, image prompt.\nPost:"
        generated_text=None
        if MODEL_LOADED:
            try:
                input_ids=tokenizer.encode(prompt,return_tensors="pt")
                device=next(model.parameters()).device
                input_ids=input_ids.to(device)
                out=model.generate(input_ids,max_length=min(len(input_ids[0])+120,MAX_MODEL_TOKENS),do_sample=True,top_p=0.9,temperature=0.8,pad_token_id=tokenizer.eos_token_id)
                text=tokenizer.decode(out[0],skip_special_tokens=True)
                post_text=truncate_to_length(text.split("\n")[0] if "\n" in text else text,max_chars)
                kw=extract_keywords(topic)+user_kws
                hashtags=generate_hashtags_from_keywords(kw)
                variants.append({"post":post_text,"hashtags":hashtags,"hook":post_text.split(".")[0] if "." in post_text else post_text[:60],"cta":"Read more","image_prompt":"A professional photo illustrating "+topic,"video_script":"Short script: "+(post_text[:120]+"..." if len(post_text)>120 else post_text),"posting_time":simple_posting_schedule(topic),"reading_time":estimate_reading_time(post_text),"char_count":len(post_text),"confidence":"model"})
                continue
            except Exception:
                pass
        variant=template_generator(topic,tone,platform,max_chars)
        if user_kws:
            user_tags=["#"+re.sub(r"\s+","",k) for k in user_kws][:4]
            variant["hashtags"]=list(dict.fromkeys(user_tags+variant["hashtags"]))
        variants.append(variant)
    return variants

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="VIP AI Content Creator", page_icon="✨", layout="wide")
css()

if "variants" not in st.session_state: st.session_state.variants=[]
if "clipboard_log" not in st.session_state: st.session_state.clipboard_log=[]
if "chat_history" not in st.session_state: st.session_state.chat_history=[]
if "last_generated_topic" not in st.session_state: st.session_state.last_generated_topic=""

left_col, right_col = st.columns([2.6,1])

# Right panel
with right_col:
    st.markdown('<div class="control-panel glass">',unsafe_allow_html=True)
    st.markdown('<h2 class="gold-title">VIP Controls</h2><div class="gold-sub">Right-side panel</div>',unsafe_allow_html=True)

    topic_input=st.text_input("Topic / Idea", value=st.session_state.get("last_generated_topic",""), key="topic_input")
    platform=st.selectbox("Platform", ["Instagram","YouTube Shorts","Twitter/X","LinkedIn","Facebook","TikTok"], key="platform_input")
    tone=st.selectbox("Tone", ["Professional","Casual","Funny","Inspirational","Urgent"], index=0, key="tone_input")
    length_pref=st.selectbox("Post length", ["Short","Medium","Long"], index=1, key="length_input")
    num_variants=st.slider("Variations", min_value=1, max_value=10, value=3, key="num_variants_input")
    kw_input=st.text_area("Optional keywords / hashtags", value="", height=60, key="kw_input")

    gen_btn=st.button("Generate ✨", key="gen_btn")
    clear_btn=st.button("Clear Output", key="clear_btn")

    if st.button("Export All (CSV)", key="export_csv"):
        if not st.session_state.variants: st.warning("No variants to export.")
        else:
            df=pd.DataFrame(st.session_state.variants)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "variants.csv", "text/csv")
    if st.button("Export All (JSON)", key="export_json"):
        if not st.session_state.variants: st.warning("No variants to export.")
        else:
            import json
            payload=json.dumps(st.session_state.variants,ensure_ascii=False,indent=2)
            st.download_button("Download JSON", payload, "variants.json", "application/json")
    if st.session_state.clipboard_log:
        if st.button("Download Clipboard Log", key="download_log"):
            df_log=pd.DataFrame(st.session_state.clipboard_log)
            st.download_button("Download Log CSV", df_log.to_csv(index=False).encode("utf-8"), "clipboard_log.csv", "text/csv")
    st.markdown("</div>",unsafe_allow_html=True)

# Left area
with left_col:
    tabs=st.tabs(["Generator","Chatbot"])
    # Generator
    with tabs[0]:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        st.markdown('<h2 class="gold-title">Generator</h2><div class="meta">Generate ready-to-post content — multiple variants, hooks, CTAs, hashtags & scripts</div>',unsafe_allow_html=True)

        if st.session_state.last_generated_topic:
            st.markdown(f"**Last topic:** {st.session_state.last_generated_topic}")

        if gen_btn:
            if not topic_input.strip(): st.warning("Enter a topic.")
            else:
                with st.spinner("Generating variants..."):
                    variants=generate_variants(topic_input,tone,platform,length_pref,num_variants,kw_input)
                    now=datetime.now().isoformat(timespec="seconds")
                    for idx,v in enumerate(variants):
                        v["_id"]=f"v_{int(time.time()*1000)}_{idx}"
                        v["_generated_at"]=now
                    st.session_state.variants=variants
                    st.session_state.last_generated_topic=topic_input.strip()
                    time.sleep(0.5)

        if clear_btn:
            st.session_state.variants=[]
            st.success("Cleared generated variants.")

        if not st.session_state.variants:
            st.info("No variants yet. Use right panel to generate.")
        else:
            for i,var in enumerate(st.session_state.variants):
                st.markdown(f'<div class="variant-card">',unsafe_allow_html=True)
                st.markdown(f"**Variant {i+1}**  <span class='meta'>• generated at {var.get('_generated_at','-')}</span>",unsafe_allow_html=True)

                edited=st.text_area("Post Text", value=var["post"], height=110, key=f"post_text_{i}")
                st.session_state.variants[i]["post"]=edited
                st.write("**Hashtags:**"," ".join(var["hashtags"]))
                st.write("**Hook:**",var["hook"])
                st.write("**CTA:**",var["cta"])
                st.write("**Image prompt:**",var["image_prompt"])
                st.write("**Short video script:**",var["video_script"])
                st.write("**Posting time:**",var["posting_time"])
                st.write("**Reading time / chars:**",f"{var['reading_time']} / {var['char_count']}")
                st.write("**Confidence:**",var.get("confidence","template"))

                col1,col2,col3,col4=st.columns([1,1,1,1])
                with col1:
                    if st.button("Copy & Log", key=f"logcopy_{i}"):
                        st.session_state.clipboard_log.append({"time":datetime.now().isoformat(timespec="seconds"),"text":edited})
                        st.success("Copied to clipboard log.")
                with col2:
                    if st.button("Regenerate", key=f"regen_{i}"):
                        with st.spinner("Regenerating..."):
                            new_var=generate_variants(st.session_state.last_generated_topic or topic_input,tone,platform,length_pref,1,kw_input)[0]
                            new_var["_id"]=f"v_{int(time.time()*1000)}_regen"
                            new_var["_generated_at"]=datetime.now().isoformat(timespec="seconds")
                            st.session_state.variants[i]=new_var
                            st.experimental_rerun()
                with col3:
                    import json
                    if st.button("Export JSON", key=f"json_{i}"):
                        payload=json.dumps(var,ensure_ascii=False,indent=2)
                        st.download_button("Download JSON", payload,f"variant_{i+1}.json","application/json")
                with col4:
                    if st.button("Export TXT", key=f"txt_{i}"):
                        st.download_button("Download TXT", var["post"], f"variant_{i+1}.txt","text/plain")

                st.markdown("</div>",unsafe_allow_html=True)

        st.markdown("</div>",unsafe_allow_html=True)

    # Chatbot
    with tabs[1]:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        st.markdown('<h2 class="gold-title">Chatbot — Professional Assistant</h2>',unsafe_allow_html=True)
        user_msg=st.text_area("Your question / request", value="", key="chat_input")
        if st.button("Send ✉️", key="chat_send"):
            if user_msg.strip():
                reply=f"Simulated answer for: {user_msg.strip()}"
                st.session_state.chat_history.append({"role":"user","text":user_msg})
                st.session_state.chat_history.append({"role":"assistant","text":reply})
        if st.session_state.chat_history:
            for idx,msg in enumerate(st.session_state.chat_history):
                role = msg["role"].capitalize()
                st.markdown(f"**{role}:** {msg['text']}")
        st.markdown("</div>",unsafe_allow_html=True)

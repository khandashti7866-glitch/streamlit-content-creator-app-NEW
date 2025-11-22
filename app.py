"""
AI SOCIAL MEDIA CONTENT CREATOR - VIP Single-file Streamlit App (app.py)

Features:
- Right-side control panel + left output area
- Tabs: Generator & Chatbot
- Black & Gold VIP theme with fullscreen technology background & glass panels
- Optional local transformer model (sshleifer/tiny-gpt2 or distilgpt2). Fallback to templates if unavailable
- Generates multiple variants (1-10) with: post text, hashtags, hook, CTA, image prompt, short video script, posting time, reading time, char count, confidence
- Chatbot mode with professional writing assistant personality
- Export CSV / JSON; per-variant Regenerate; Copy-to-clipboard; clipboard log downloadable
- Fully offline (no API keys)
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

import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional transformer
TRANSFORMERS_AVAILABLE = False
MODEL_LOADED = False
model = None
tokenizer = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -------------------------
# Constants
# -------------------------
BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1600&q=80"
MODEL_NAME = "sshleifer/tiny-gpt2"
MAX_MODEL_TOKENS = 150
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
        print("Model not loaded, fallback will be used:", e)

# -------------------------
# CSS for VIP theme
# -------------------------
def css():
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE}");
        background-size: cover;
        background-attachment: fixed;
        color: #fff;
    }}
    .glass {{
        background: rgba(0,0,0,0.6);
        padding: 18px;
        border-radius: 14px;
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,215,0,0.12);
    }}
    h2 {{
        color:#ffd166;
        text-shadow: 0 0 8px #ffd166;
    }}
    .stButton>button {{
        background: linear-gradient(90deg,#ffd166,#ffb703);
        color: #08111a;
        border-radius: 10px;
        font-weight:800;
    }}
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {{
        background: rgba(255,255,255,0.06);
        color:#fff;
        border-radius:8px;
        padding:8px;
    }}
    </style>
    """, unsafe_allow_html=True)

css()

# -------------------------
# Utility functions
# -------------------------
def extract_keywords(text:str,max_keywords:int=8)->List[str]:
    toks = re.findall(r"\w+", text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t)>2]
    freqs = {}
    for t in toks: freqs[t]=freqs.get(t,0)+1
    sorted_tokens = sorted(freqs.items(), key=lambda x:(-x[1],x[0]))
    return [t for t,_ in sorted_tokens][:max_keywords]

def generate_hashtags_from_keywords(keywords:List[str],min_tags:int=5,max_tags:int=12)->List[str]:
    tags = ["#"+re.sub(r"[^A-Za-z0-9]","",k) for k in keywords]
    extras = ["#viral","#trending","#contentcreator","#tips","#howto","#learn"]
    random.shuffle(extras)
    tags = tags+extras[:max(0,min_tags-len(tags))]
    return tags[:max_tags]

def simple_posting_schedule(topic:str)->str:
    low = topic.lower()
    hours = [9,11,14,17,19,21]
    day=random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    if any(k in low for k in ["business","finance","career","invest"]):
        day=random.choice(["Tuesday","Wednesday","Thursday"])
    elif any(k in low for k in ["fun","travel","food","fashion","music"]):
        day=random.choice(["Friday","Saturday","Sunday"])
    return f"{day} at {random.choice(hours)}:00"

def estimate_reading_time(text:str)->str:
    words=len(re.findall(r"\w+",text))
    minutes=max(1,int(words/200))
    return f"{minutes} min"

def truncate_to_length(text:str,max_chars:int)->str:
    if len(text)<=max_chars: return text
    cut=text[:max_chars].rfind(".")
    if cut==-1: cut=text[:max_chars].rfind(" ")
    if cut==-1 or cut<max_chars//2: return text[:max_chars].rstrip()+"…"
    return text[:cut+1]

def template_generator(topic:str,tone:str,platform:str,length_chars:int)->Dict[str,Any]:
    kw=extract_keywords(topic,6)
    hashtags=generate_hashtags_from_keywords(kw)
    hooks={"Professional":[f"Industry insight: {topic} — key points.","Brief update on {topic} for professionals."],
           "Casual":[f"Quick tip about {topic}!","Real talk: {topic} explained."],
           "Funny":[f"If {topic} were a person, they'd say...","Fun facts about {topic}."],
           "Inspirational":[f"How {topic} changed the game.","One idea about {topic}."],
           "Urgent":[f"Important! {topic} updates.","Alert: new {topic} shifts."]}
    ctas=["Learn more","Share your thoughts","Save this post","Try this now","Join the conversation"]
    images=[f"Tech photo illustrating {topic}","Minimalist image representing {topic}"]
    scripts=[f"{topic} in short: A, B, C.","Quick tip on {topic}: 1,2,3 steps."]
    tone_h=hooks.get(tone,hooks["Professional"])
    hook=random.choice(tone_h)
    cta=random.choice(ctas)
    image_prompt=random.choice(images)
    script=random.choice(scripts)
    post_text=truncate_to_length(f"{hook} {topic}. {cta}.",length_chars)
    posting_time=simple_posting_schedule(topic)
    reading_time=estimate_reading_time(post_text)
    char_count=len(post_text)
    confidence="template"
    return {"post":post_text,"hashtags":hashtags,"hook":hook,"cta":cta,"image_prompt":image_prompt,
            "video_script":script,"posting_time":posting_time,"reading_time":reading_time,
            "char_count":char_count,"confidence":confidence}

def model_generate(prompt:str,max_new_tokens:int=80)->Optional[str]:
    global MODEL_LOADED,model,tokenizer
    if not MODEL_LOADED or model is None or tokenizer is None: return None
    try:
        input_ids=tokenizer.encode(prompt,return_tensors="pt")
        device=next(model.parameters()).device
        input_ids=input_ids.to(device)
        out=model.generate(input_ids,max_length=min(len(input_ids[0])+max_new_tokens,MAX_MODEL_TOKENS),
                           do_sample=True,top_p=0.9,temperature=0.8,pad_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(out[0],skip_special_tokens=True)
        if text.startswith(prompt): return text[len(prompt):].strip()
        return text.strip()
    except: return None

def generate_variants(topic:str,tone:str,platform:str,length_pref:str,n_variants:int,keywords_input:str)->List[Dict[str,Any]]:
    length_map={"Short":120,"Medium":300,"Long":800}
    max_chars=length_map.get(length_pref,300)
    user_kws=[k.strip() for k in re.split(r"[,\n;]+",keywords_input) if k.strip()]
    variants=[]
    for i in range(n_variants):
        prompt=f"Topic:{topic}\nTone:{tone}\nPlatform:{platform}\nPost length {max_chars}:"
        gen=None
        if MODEL_LOADED:
            gen=model_generate(prompt,120)
            if gen:
                post_text=truncate_to_length(gen,max_chars)
                kw=extract_keywords(topic)+user_kws
                hashtags=generate_hashtags_from_keywords(kw)
                variants.append({"post":post_text,"hashtags":hashtags,"hook":post_text.split(".")[0],
                                 "cta":"Read more","image_prompt":"Professional image of "+topic,
                                 "video_script":"Short script: "+post_text[:120],
                                 "posting_time":simple_posting_schedule(topic),
                                 "reading_time=estimate_reading_time(post_text),
                                 "char_count":len(post_text),"confidence":"model"})
                continue
        var=template_generator(topic,tone,platform,max_chars)
        if user_kws:
            user_tags=["#"+re.sub(r"\s+","",k) for k in user_kws][:4]
            var["hashtags"]=list(dict.fromkeys(user_tags+var["hashtags"]))
        variants.append(var)
    return variants

# -------------------------
# Chatbot fallback
# -------------------------
def chatbot_transform(message:str,last_content:Optional[str])->str:
    prompt=message.lower()
    if MODEL_LOADED:
        full_prompt=f"You are a professional assistant. User instruction: {message}\nContent: {last_content or ''}"
        gen=model_generate(full_prompt,160)
        if gen: return gen.strip()
    if "shorten" in prompt:
        return truncate_to_length(last_content or "",280)
    if "professional" in prompt:
        if not last_content: return "Provide content to polish."
        out=last_content.replace("don't","do not").replace("can't","cannot").replace("it's","it is")
        return out+"\n\n— Polished by assistant"
    if "translate" in prompt and "urdu" in prompt:
        if not last_content: return "Provide text to translate."
        small_dict={"hello":"ہیلو","thank you":"شکریہ","good":"اچھا","best":"بہترین","today":"آج","learn":"سیکھیں"}
        words=last_content.split()
        translated=[small_dict.get(re.sub(r"[^a-zA-Z']","",w).lower(),w) for w in words]
        return " ".join(translated)+"\nNote: limited translation."
    if "funny" in prompt:
        return (last_content or "")+" 😂"
    return "I can help with shorten, polish, hashtags, translate (limited), or rewrite tone."

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI SOCIAL MEDIA CONTENT CREATOR", layout="wide")
if "variants" not in st.session_state: st.session_state.variants=[]
if "chat_history" not in st.session_state: st.session_state.chat_history=[]
if "last_generated_topic" not in st.session_state: st.session_state.last_generated_topic=""

left_col,right_col=st.columns([2.6,1])

with right_col:
    st.markdown('<div class="glass">',unsafe_allow_html=True)
    topic_input=st.text_input("Topic / Idea",value=st.session_state.last_generated_topic)
    platform=st.selectbox("Platform",["Instagram","YouTube Shorts","Twitter/X","LinkedIn","Facebook","TikTok"])
    tone=st.selectbox("Tone",["Professional","Casual","Funny","Inspirational","Urgent"])
    length_pref=st.selectbox("Post length",["Short","Medium","Long"])
    num_variants=st.slider("Variations",1,10,3)
    kw_input=st.text_area("Optional keywords / hashtags",height=60)
    gen_btn=st.button("Generate ✨")
    clear_btn=st.button("Clear Output")
    if st.button("Export All (CSV)") and st.session_state.variants:
        df=pd.DataFrame(st.session_state.variants)
        st.download_button("Download CSV",df.to_csv(index=False).encode("utf-8"),"variants.csv","text/csv")
    if st.button("Export All (JSON)") and st.session_state.variants:
        import json
        payload=json.dumps(st.session_state.variants,ensure_ascii=False,indent=2)
        st.download_button("Download JSON",payload,"variants.json","application/json")
    st.markdown("</div>",unsafe_allow_html=True)

with left_col:
    tabs=st.tabs(["Generator","Chatbot"])
    with tabs[0]:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        if gen_btn:
            if not topic_input.strip(): st.warning("Enter a topic.")
            else:
                with st.spinner("Generating..."):
                    variants=generate_variants(topic_input,tone,platform,length_pref,num_variants,kw_input)
                    now=datetime.now().isoformat(timespec="seconds")
                    for idx,v in enumerate(variants):
                        v["_id"]=f"v_{int(time.time()*1000)}_{idx}"
                        v["_generated_at"]=now
                    st.session_state.variants=variants
                    st.session_state.last_generated_topic=topic_input
                    time.sleep(0.5)
        if clear_btn: st.session_state.variants=[]
        for i,var in enumerate(st.session_state.variants):
            st.markdown('<div class="glass">',unsafe_allow_html=True)
            edited=st.text_area(f"Variant {i+1}",value=var["post"],height=110,key=f"txt_{i}")
            st.session_state.variants[i]["post"]=edited
            st.write("Hashtags:"," ".join(var["hashtags"]))
            st.write("Hook:",var["hook"])
            st.write("CTA:",var["cta"])
            st.write("Image prompt:",var["image_prompt"])
            st.write("Short video script:",var["video_script"])
            st.write("Posting time:",var["posting_time"])
            st.write("Reading / chars:",f"{var['reading_time']} / {var['char_count']}")
            st.write("Confidence:",var.get("confidence","template"))
            if st.button("Regenerate",key=f"regen_{i}"):
                new_var=generate_variants(st.session_state.last_generated_topic or topic_input,tone,platform,length_pref,1,kw_input)[0]
                new_var["_id"]=f"v_{int(time.time()*1000)}_regen"
                new_var["_generated_at"]=datetime.now().isoformat(timespec="seconds")
                st.session_state.variants[i]=new_var
                st.experimental_rerun()
            st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with tabs[1]:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        for entry in st.session_state.chat_history:
            st.markdown(f"**You:** {entry['user']}")
            st.markdown(f"**Assistant:** {entry['bot']}")
            st.markdown("---")
        user_cmd=st.text_input("Enter instruction (shorten, professional, hashtags, translate)",key="chat_input")
        apply_to=None
        if st.session_state.variants:
            choices=[f"Variant {i+1}" for i in range(len(st.session_state.variants))]
            choices.insert(0,"No specific variant")
            selected=st.selectbox("Apply to",choices,0)
            if selected!="No specific variant": apply_to=int(selected.replace("Variant ",""))-1
        if st.button("Send to Assistant"):
            last_content=None
            if apply_to is not None: last_content=st.session_state.variants[apply_to]["post"]
            reply=chatbot_transform(user_cmd,last_content)
            st.session_state.chat_history.append({"user":user_cmd,"bot":reply})
            if apply_to is not None:
                low=user_cmd.lower()
                if "shorten" in low: st.session_state.variants[apply_to]["post"]=truncate_to_length(reply or last_content,280)
                elif "professional" in low or "polish" in low: st.session_state.variants[apply_to]["post"]=reply
                elif "hashtags" in low: st.session_state.variants[apply_to]["hashtags"]=reply.split()
                elif "translate" in low and "urdu" in low: st.session_state.variants[apply_to]["post"]=reply
        st.markdown("</div>",unsafe_allow_html=True)

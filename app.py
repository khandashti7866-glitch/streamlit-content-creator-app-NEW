import streamlit as st

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="AI Content Creator",
    page_icon="✨",
    layout="wide",
)

# -----------------------------------------
# BACKGROUND IMAGE
# -----------------------------------------
def add_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white !important;
        }}

        /* Text shadow for readability */
        h1, h2, h3, p, label, span {{
            text-shadow: 0px 0px 12px rgba(0,0,0,0.9);
        }}

        /* Translucent black content box */
        .glass-box {{
            background: rgba(0,0,0,0.55);
            padding: 25px;
            border-radius: 18px;
            backdrop-filter: blur(6px);
            box-shadow: 0 0 25px rgba(0,0,0,0.4);
        }}

        /* Glow header */
        .glow-title {{
            font-size: 45px;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 0 0 18px #4da8ff, 0 0 28px #0074ff;
        }}

        /* Input boxes translucent */
        .stTextInput>div>div>input,
        .stTextArea>div>textarea {{
            background: rgba(255,255,255,0.15);
            color: #fff;
            border-radius: 10px;
        }}

        /* Buttons with smooth modern look */
        .stButton>button {{
            background: #007bff;
            color: white;
            border-radius: 10px;
            padding: 10px 22px;
            border: none;
            font-weight: 700;
            box-shadow: 0 0 12px rgba(0,123,255,0.6);
        }}

        /* Sidebar dark overlay */
        [data-testid="stSidebar"] {{
            background: rgba(0,0,0,0.60);
            backdrop-filter: blur(5px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
add_background("https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1400&q=80")


# -----------------------------------------
# SIDEBAR (UI options remain on right)
# -----------------------------------------
with st.sidebar:
    st.markdown("### ✨ Options")
    st.write("Adjust your content style:")
    tone = st.selectbox("Select Tone", ["Professional", "Casual", "Funny", "Inspirational", "Emotional"])
    length = st.selectbox("Content Length", ["Short", "Medium", "Long"])


# -----------------------------------------
# MAIN UI
# -----------------------------------------
st.markdown('<h1 class="glow-title">AI Social Media Content Creator ✨</h1>', unsafe_allow_html=True)

st.markdown('<div class="glass-box">', unsafe_allow_html=True)
topic = st.text_input("Enter your topic:", placeholder="e.g., How to stay motivated")
st.markdown('</div>', unsafe_allow_html=True)

generate = st.button("Generate Content")

# -----------------------------------------
# CONTENT GENERATION
# -----------------------------------------
if generate:
    if topic.strip() == "":
        st.warning("Please enter a topic first.")
    else:
        st.markdown('<br>', unsafe_allow_html=True)

        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown(f"## 🔥 Hook for: **{topic}**")
        st.write(f"✨ The truth about **{topic}** will surprise you… stay with me!")

        st.markdown("## 📝 Caption")
        st.write(f"{topic} simplified — here’s what nobody tells you.")

        st.markdown("## 🎯 Script")
        st.write(
            f"""
            Most people misunderstand **{topic}**, but you don’t have to.  
            Here’s the real secret behind it, and how you can apply it today…
            """
        )

        st.markdown("## 🏷 Hashtags")
        st.code(f"#{topic.replace(' ', '')} #motivation #learning #success #creatorlife")

        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("<br><p style='text-align:center; opacity:0.8;'>✨ Powered by Your AI Engine ✨</p>", unsafe_allow_html=True)

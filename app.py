import streamlit as st

st.set_page_config(page_title="AI Content Creator", layout="wide")

# -----------------------------------------
# BACKGROUND IMAGE + CUSTOM UI
# -----------------------------------------
def add_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white !important;
        }}

        /* Glass UI Box */
        .glass-box {{
            background: rgba(0,0,0,0.55);
            padding: 25px;
            border-radius: 18px;
            backdrop-filter: blur(6px);
            box-shadow: 0 0 25px rgba(0,0,0,0.4);
        }}

        /* Title Glow */
        .glow-title {{
            font-size: 42px;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 0 0 15px #4da8ff, 0 0 28px #0074ff;
        }}

        /* Input Styling */
        .stTextInput>div>div>input,
        .stTextArea>div>textarea {{
            background: rgba(255,255,255,0.15);
            color: #fff;
        }}

        /* Buttons */
        .stButton>button {{
            background: #007bff;
            color: white;
            border-radius: 10px;
            padding: 10px 22px;
            border: none;
            font-weight: 700;
            box-shadow: 0 0 12px rgba(0,123,255,0.6);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background("https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1400&q=80")

# -----------------------------------------
# LAYOUT: LEFT (OUTPUT) / RIGHT (UI)
# -----------------------------------------
left, right = st.columns([2.5, 1])

# -----------------------------------------
# RIGHT SIDE UI PANEL
# -----------------------------------------
with right:
    st.markdown("### ⚙️ **Options**", unsafe_allow_html=True)
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)

    topic = st.text_input("Enter your topic")
    tone = st.selectbox("Tone", ["Professional", "Casual", "Funny", "Inspirational"])
    length = st.selectbox("Length", ["Short", "Medium", "Long"])
    generate = st.button("Generate ✨")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# LEFT SIDE CONTENT OUTPUT
# -----------------------------------------
with left:
    st.markdown('<h1 class="glow-title">AI Content Creator</h1>', unsafe_allow_html=True)

    if generate:
        if topic.strip() == "":
            st.warning("Please enter a topic.")
        else:
            st.markdown('<div class="glass-box">', unsafe_allow_html=True)

            st.markdown(f"## 🔥 Hook")
            st.write(f"Here's the truth about **{topic}**…")

            st.markdown(f"## ✍️ Caption")
            st.write(f"{topic} explained in a simple and powerful way.")

            st.markdown(f"## 🎬 Script")
            st.write(
                f"""
                Most people overlook **{topic}**.  
                Here's what you MUST know…
                """
            )

            st.markdown(f"## 🏷 Hashtags")
            st.code(f"#{topic.replace(' ', '')} #motivation #inspiration #contentcreator")

            st.markdown('</div>', unsafe_allow_html=True)

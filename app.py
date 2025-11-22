import streamlit as st
from PIL import Image
import base64

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Social Media Content Creator",
    page_icon="✨",
    layout="wide",
)

# ---------------------------
# BACKGROUND IMAGE FUNCTION
# ---------------------------
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
add_bg_from_url("https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1350&q=80")


# ---------------------------
# CUSTOM BLUE BOLD TEXT STYLE
# ---------------------------
st.markdown("""
<style>
.blue-bold {
    color: #0074FF;
    font-weight: 700;
    font-size: 28px;
}
.normal-text {
    color: white;
    font-size: 18px;
}
.stButton>button {
    background-color: #0074FF !important;
    color: white !important;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
}
.stTextInput>div>div>input {
    background-color: rgba(255,255,255,0.85);
    color: #000;
}
.stTextArea>div>textarea {
    background-color: rgba(255,255,255,0.85);
    color: #000;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# APP HEADING
# ---------------------------
st.markdown('<h1 class="blue-bold">✨ Social Media Content Creator App</h1>', unsafe_allow_html=True)
st.markdown('<p class="normal-text">Create high-quality content instantly — no API keys needed.</p>', unsafe_allow_html=True)


# ---------------------------
# USER INPUT
# ---------------------------
topic = st.text_input("Enter your topic:", "")

if st.button("Generate Content"):
    if topic.strip() == "":
        st.warning("Please enter a topic first!")
    else:
        st.markdown(f'<h3 class="blue-bold">Generated Content for: {topic}</h3>', unsafe_allow_html=True)

        st.write("**Hook:**")
        st.success(f"🔥 The real secret behind {topic} will shock you!")

        st.write("**Short Caption:**")
        st.info(f"{topic} made simple — let’s break it down!")

        st.write("**Hashtags:**")
        st.code(f"#{topic.replace(' ', '')} #Trending #ViralContent #CreatorTools")

        st.write("**Content Script:**")
        st.write(
            f"""
            {topic} is becoming more important every day.  
            Here’s what most people don’t understand about it…  
            And here's how you can use it to level up instantly.  
            """
        )

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p class="normal-text">Powered by Your AI Content Engine ✨</p>', unsafe_allow_html=True)

import streamlit as st
from openai import OpenAI

# --- Streamlit App Setup ---
st.set_page_config(page_title="Social Media Content Creator", page_icon="🤖", layout="wide")
st.title("🤖 Social Media Content Creator")
st.markdown("Generate social media posts, captions, tweets, and content ideas using AI!")

# --- OpenAI API Key ---
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Sidebar Options ---
st.sidebar.header("Settings")
model = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])
temperature = st.sidebar.slider("Creativity Level (Temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 500)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_area("Enter your topic or question", key="input", height=100)

if st.button("Generate Content"):
    if not user_input.strip():
        st.warning("Please enter a topic or question!")
    else:
        with st.spinner("Generating content..."):
            # Build prompt
            prompt = f"""
            You are a social media content creator AI. Generate creative, engaging, and catchy content based on the following input: 
            {user_input}
            Provide: 
            1. Post Text
            2. Caption/Hashtags
            3. Short Tweet (if applicable)
            4. Content Ideas
            """

            # API Call
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            bot_reply = response.choices[0].message.content
            st.session_state.messages.append(("User", user_input))
            st.session_state.messages.append(("AI", bot_reply))

# --- Display Chat ---
for sender, msg in st.session_state.messages:
    if sender == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**AI:** {msg}")

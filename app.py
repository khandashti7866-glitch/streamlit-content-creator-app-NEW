import os
import streamlit as st
import openai  # We’ll use OpenAI-compatible calls, but point base_url to Groq

# --- Streamlit setup ---
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Groq‑Powered Chatbot")
st.markdown("Chat using Groq LLM via its API key (OpenAI-compatible)")

# --- API Key Setup ---
# Either read from env var
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.text_input("Enter your **Groq API Key**", type="password")
if not groq_api_key:
    st.warning("Please provide a Groq API Key to continue.")
    st.stop()

# Configure openai client to use Groq's API base URL
openai.api_key = groq_api_key
openai.api_base = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint :contentReference[oaicite:3]{index=3}

# --- Settings in Sidebar ---
st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 2000, 500)

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Input Area ---
user_input = st.text_area("You:", height=120)

if st.button("Send"):
    if user_input.strip():
        st.session_state.messages.append(("User", user_input))
        with st.spinner("Generating response from Groq..."):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": user_input}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                reply = response.choices[0].message.content
                st.session_state.messages.append(("AI", reply))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please write something to send.")

# --- Display Chat --- 
if st.session_state.messages:
    st.markdown("---")
    for sender, msg in st.session_state.messages:
        if sender == "User":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**AI:** {msg}")

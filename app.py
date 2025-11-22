import streamlit as st
from transformers import pipeline, set_seed

# --- Streamlit Setup ---
st.set_page_config(page_title="Offline Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Social Media Content Creator (Offline)")
st.markdown("This chatbot runs locally and does NOT require any API key.")

# --- Initialize Local Model ---
@st.cache_resource(show_spinner=True)
def load_model():
    generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)  # set device=-1 for CPU
    set_seed(42)
    return generator

generator = load_model()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- User Input ---
user_input = st.text_area("Enter your topic or question", key="input", height=100)

# --- Generate Response ---
if st.button("Generate Content"):
    if not user_input.strip():
        st.warning("Please enter a topic or question!")
    else:
        with st.spinner("Generating content locally..."):
            prompt = f"""
            You are a social media content creator AI.
            Generate creative social media posts, captions, tweets, and content ideas for: {user_input}
            """
            try:
                result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
                bot_reply = result[0]['generated_text']

                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("AI", bot_reply))

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

# --- Display Chat History ---
if st.session_state.messages:
    st.markdown("---")
    for sender, msg in st.session_state.messages:
        if sender == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**AI:** {msg}")
            st.download_button(
                label="📄 Copy AI Response",
                data=msg,
                file_name="ai_content.txt",
                mime="text/plain",
                key=f"download_{len(msg)}"
            )

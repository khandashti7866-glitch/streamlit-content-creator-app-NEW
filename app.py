import streamlit as st
from openai import OpenAI
from openai.error import RateLimitError, APIError, OpenAIError

# --- Streamlit App Setup ---
st.set_page_config(page_title="Social Media Content Creator", page_icon="🤖", layout="wide")
st.title("🤖 Social Media Content Creator")
st.markdown(
    "Generate social media posts, captions, tweets, and content ideas with AI!\n\n"
    "💡 Enter a topic or idea, and let the AI create engaging content for you."
)

# --- OpenAI API Key Input ---
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Sidebar Settings ---
st.sidebar.header("Settings")
model = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])
temperature = st.sidebar.slider("Creativity Level (Temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, 500)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- User Input ---
user_input = st.text_area("Enter your topic or question", key="input", height=100)

# --- Generate Content ---
if st.button("Generate Content"):
    if not user_input.strip():
        st.warning("Please enter a topic or question!")
    else:
        with st.spinner("Generating content..."):
            prompt = f"""
            You are a professional social media content creator AI.
            Generate creative, engaging, and catchy content based on the following input: 
            {user_input}

            Provide:
            1. Post Text
            2. Caption/Hashtags
            3. Short Tweet (if applicable)
            4. Additional Content Ideas
            Make it suitable for social media engagement.
            """

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                bot_reply = response.choices[0].message.content

                # Save messages to session
                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("AI", bot_reply))

            except RateLimitError:
                st.error("⚠️ Rate limit exceeded. Please wait a few seconds or check your OpenAI plan.")
            except APIError as e:
                st.error(f"⚠️ OpenAI API Error: {e}")
            except OpenAIError as e:
                st.error(f"⚠️ An OpenAI error occurred: {e}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")

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

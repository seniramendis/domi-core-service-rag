import streamlit as st
from domi.engine import DomiEngine

# Page Configuration
st.set_page_config(page_title="Domi Core AI", page_icon="⚡", layout="centered")

# Header & Branding
st.title("⚡ Domi Enterprise Assistant")
st.caption("Powered by **Dopmin Intelligence Platform** | *Air-Gapped & Offline*")
st.divider()


@st.cache_resource
def load_engine(model_name: str = "llama3"):
    return DomiEngine(model_name=model_name)


with st.spinner("Initializing Core Engine... (This may take a minute on first boot)"):
    domi = load_engine("llama3")

# Session State Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings. I am Domi. How can I assist with your enterprise operations today?"}
    ]

# Render Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_prompt := st.chat_input("Ask Domi about company policies, support hours, or fees..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        streamed_chunks = []

        def stream_with_capture(question: str):
            for chunk in domi.stream_query(question):
                streamed_chunks.append(chunk)
                yield chunk

        st.write_stream(stream_with_capture(user_prompt))
        answer = "".join(streamed_chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer})
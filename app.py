import streamlit as st
from domi.engine import DomiEngine

# Page Configuration
st.set_page_config(page_title="Domi Core AI", page_icon="⚡", layout="centered")

# Header & Branding
st.title("⚡ Domi Enterprise Assistant")
st.caption("Powered by **Dopmin Intelligence Platform** | *Air-Gapped & Offline*")
st.divider()

# Cache the engine initialization
@st.cache_resource
def get_domi():
    return DomiEngine()

with st.spinner("Initializing Core Engine... (This may take a minute on first boot)"):
    domi = get_domi()

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
        with st.spinner("Analyzing knowledge base..."):
            answer = domi.query(user_prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
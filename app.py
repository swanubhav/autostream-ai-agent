import streamlit as st
from graph import graph

st.set_page_config(page_title="AutoStream AI", layout="centered")

st.title("🎬 AutoStream AI Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "intent": None,
        "name": None,
        "email": None,
        "platform": None,
        "response": None
    }

# Display chat
for i, msg in enumerate(st.session_state.messages):
    role = "user" if i % 2 == 0 else "assistant"
    st.chat_message(role).write(msg)

# Input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(user_input)

    # update agent state
    st.session_state.agent_state["messages"].append(user_input)

    result = graph.invoke(st.session_state.agent_state)

    response = result.get("response", "No response generated")

    st.session_state.messages.append(response)

    st.session_state.agent_state = result

    st.chat_message("assistant").write(response)
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

from state import AgentState
from rag import RAG
from tools import mock_lead_capture

# -----------------------------
# INITIALIZE LLM (Gemini)
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

rag = RAG()

# -----------------------------
# INTENT DETECTION NODE
# -----------------------------
def detect_intent(state: AgentState):
    user_input = state["messages"][-1]

    prompt = f"""
    Classify the user's intent into ONE word only:
    greeting, pricing, or high_intent.

    Examples:
    Hi → greeting
    What are your pricing plans → pricing
    I want to try Pro plan → high_intent

    Input: {user_input}
    """

    result = llm.invoke(prompt)
    text = result.content.lower().strip()

    if "greeting" in text:
        state["intent"] = "greeting"
    elif "pricing" in text:
        state["intent"] = "pricing"
    elif "high" in text:
        state["intent"] = "high_intent"
    else:
        state["intent"] = "general"

    return state


# -----------------------------
# RESPONSE NODE (RAG + GENERAL)
# -----------------------------
def generate_response(state: AgentState):
    user_input = state["messages"][-1]
    intent = state.get("intent")

    # Greeting
    if intent == "greeting":
        state["response"] = "Hey! 👋 I can help you with AutoStream pricing, features, or getting started."
        return state

    # Pricing → RAG
    if intent == "pricing":
        context = rag.retrieve(user_input)

        result = llm.invoke(f"""
        Use the context below to answer clearly and concisely.

        Context:
        {context}

        Question:
        {user_input}
        """)

        state["response"] = result.content
        return state

    # Fallback
    state["response"] = "I can help with pricing or getting started. What would you like to know?"
    return state


# -----------------------------
# LEAD COLLECTION NODE
# -----------------------------
def lead_collection(state: AgentState):
    user_input = state["messages"][-1]

    # Ask for name
    if not state.get("name"):
        state["name"] = user_input
        state["response"] = "Great! Let's get you started. What's your name?"
        return state

    # Ask for email
    elif not state.get("email"):
        state["email"] = user_input
        state["response"] = "What's your email?"
        return state

    # Ask for platform + trigger tool
    elif not state.get("platform"):
        state["platform"] = user_input

        result = mock_lead_capture(
            state["name"],
            state["email"],
            state["platform"]
        )

        state["response"] = result
        return state

    return state


# -----------------------------
# ROUTING LOGIC
# -----------------------------
def route(state: AgentState):
    intent = state.get("intent")

    if intent == "high_intent":
        return "lead"

    return "response"


# -----------------------------
# BUILD GRAPH
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("intent", detect_intent)
builder.add_node("response", generate_response)
builder.add_node("lead", lead_collection)

# Entry point
builder.set_entry_point("intent")

# Conditional routing
builder.add_conditional_edges(
    "intent",
    route,
    {
        "response": "response",
        "lead": "lead"
    }
)

# End nodes
builder.add_edge("response", END)
builder.add_edge("lead", END)

# Compile graph
graph = builder.compile()
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from state import AgentState
from rag import RAG
from tools import mock_lead_capture

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key
)

rag = RAG()

# -----------------------------
# INTENT DETECTION NODE
# -----------------------------
def detect_intent(state: AgentState):
    user_input = state["messages"][-1]

    prompt = f"""
    Classify the user's intent into ONE of:
    - greeting
    - pricing
    - high_intent

    Examples:
    Hi → greeting
    What are your plans → pricing
    I want to buy → high_intent

    Input: {user_input}
    Answer ONLY one word.
    """

    result = llm.invoke(prompt).content.lower().strip()

    if "greeting" in result:
        state["intent"] = "greeting"
    elif "pricing" in result:
        state["intent"] = "pricing"
    elif "high" in result:
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

        response = llm.invoke(f"""
        Use the context below to answer the user clearly.

        Context:
        {context}

        Question:
        {user_input}
        """).content

        state["response"] = response
        return state

    # General fallback
    state["response"] = "I can help with pricing or getting started. What would you like to know?"
    return state


# -----------------------------
# LEAD COLLECTION NODE
# -----------------------------
def lead_collection(state: AgentState):
    user_input = state["messages"][-1]

    # Ask name
    if not state.get("name"):
        state["name"] = user_input
        state["response"] = "Great! What's your email?"
        return state

    # Ask email
    elif not state.get("email"):
        state["email"] = user_input
        state["response"] = "Which platform do you create content on? (YouTube/Instagram)"
        return state

    # Ask platform and trigger tool
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

    # If user is ready → go to lead capture
    if intent == "high_intent":
        return "lead"

    # Otherwise → normal response
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

# Conditional routing (IMPORTANT FIX)
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
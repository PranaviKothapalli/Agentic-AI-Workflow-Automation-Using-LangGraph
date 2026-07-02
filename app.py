import os
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini LLM (Automatically reads GEMINI_API_KEY from environment)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)

# -----------------------------
# Graph State Definition
# -----------------------------
class GraphState(TypedDict):
    user_query: str
    retrieved_docs: str
    draft_response: str
    critic_feedback: str
    retry_count: int
    is_valid: bool


# -----------------------------
# 1. Research Agent Node
# -----------------------------
def research_agent(state: GraphState):
    print("\n[Research Agent] Searching knowledge base...")
    query = state["user_query"]

    # In production, replace this with your actual ChromaDB / FAISS vector database lookup
    context = (
        f"Context Asset: Agentic AI frameworks use LangGraph to build stateful multi-agent systems. "
        f"Unlike chains, graphs allow cycles, conditional routing, and iterative state updates. "
        f"Key concepts include Nodes (functions), Edges (routing), and State (shared memory memory-dict)."
    )

    return {
        "retrieved_docs": context
    }


# -----------------------------
# 2. Analysis Agent Node
# -----------------------------
def analysis_agent(state: GraphState):
    print("[Analysis Agent] Generating response based on query and context...")

    system_prompt = f"""
You are an expert AI Analysis Agent. Your task is to provide a comprehensive answer to the user's query.
You MUST strictly use the provided context below. If previous critic feedback is present, address the mistakes mentioned.

Context:
{state['retrieved_docs']}

Previous Critique/Feedback:
{state.get('critic_feedback', 'None')}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_query"])
    ]

    response = llm.invoke(messages)

    return {
        "draft_response": response.content
    }


# -----------------------------
# Validation Schema (Pydantic)
# -----------------------------
class ValidationOutput(BaseModel):
    is_valid: bool = Field(
        description="Set to True ONLY if the response directly answers the query accurately using the context. Set to False if missing detail, incorrect, or hallucinated."
    )
    feedback: str = Field(
        description="Detailed suggestions or criticisms if the response is incorrect or needs revision. Empty string if valid."
    )


# -----------------------------
# 3. Validation Agent Node
# -----------------------------
def validation_agent(state: GraphState):
    print("[Validation Agent] Reviewing generated output compliance...")

    # Force the Gemini model to respond matching the Pydantic schema structure
    validator = llm.with_structured_output(ValidationOutput)

    system_prompt = (
        "You are a strict Quality Assurance Critic. Evaluate the generated response against the user's query. "
        "Ensure technical accuracy and alignment with the context. Be critical of vague answers."
    )

    user_content = f"""
User Query:
{state['user_query']}

Generated Response to Review:
{state['draft_response']}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]

    result = validator.invoke(messages)
    
    # Track current iteration attempt count securely
    current_retry = state.get("retry_count", 0) + 1
    print(f"-> Validation Result | Is Valid: {result.is_valid} | Current Loop Attempt: {current_retry}/3")

    return {
        "is_valid": result.is_valid,
        "critic_feedback": result.feedback,
        "retry_count": current_retry
    }


# -----------------------------
# Conditional Routing Logic
# -----------------------------
def router(state: GraphState):
    if state["is_valid"]:
        print("[Router] Response Approved! Exiting graph workflow.")
        return "approved"

    if state["retry_count"] >= 3:
        print("[Router] Maximum loop retry threshold (3) reached. Fallback termination initiated.")
        return "approved"

    print("[Router] Response rejected by critic. Routing back to Analysis Agent for refinement.")
    return "retry"


# -----------------------------
# Graph Construction
# -----------------------------
workflow = StateGraph(GraphState)

# Register workflow nodes
workflow.add_node("ResearchAgent", research_agent)
workflow.add_node("AnalysisAgent", analysis_agent)
workflow.add_node("ValidationAgent", validation_agent)

# Set network graph entry configuration
workflow.set_entry_point("ResearchAgent")

# Connect standard structural paths
workflow.add_edge("ResearchAgent", "AnalysisAgent")
workflow.add_edge("AnalysisAgent", "ValidationAgent")

# Establish conditional state evaluation edges
workflow.add_conditional_edges(
    "ValidationAgent",
    router,
    {
        "approved": END,
        "retry": "AnalysisAgent"
    }
)

# Compile functional application layout
app = workflow.compile()


# -----------------------------
# Application Execution Execution
# -----------------------------
if __name__ == "__main__":
    initial_state = {
        "user_query": "Explain how Agentic AI operates within stateful production graph structures using LangGraph.",
        "retrieved_docs": "",
        "draft_response": "",
        "critic_feedback": "",
        "retry_count": 0,
        "is_valid": False
    }

    print("\n========== Initializing Agentic AI Loop Workflow ==========")
    
    final_output = app.invoke(initial_state)

    print("\n========== Final System Response ==========\n")
    print(final_output["draft_response"])
    print("\n===========================================\n")
import os
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# Graph State
# -----------------------------
class GraphState(TypedDict):
    user_query: str
    retrieved_docs: str
    draft_response: str
    critic_feedback: str
    retry_count: int
    is_valid: bool


# -----------------------------
# Research Agent
# -----------------------------
def research_agent(state: GraphState):

    print("\n[Research Agent] Searching knowledge base...")

    query = state["user_query"]

    # Mock retrieval (Replace with ChromaDB if needed)
    context = (
        f"Retrieved information related to: '{query}'. "
        "This information will be used to generate a better response."
    )

    return {
        "retrieved_docs": context
    }


# -----------------------------
# Analysis Agent
# -----------------------------
def analysis_agent(state: GraphState):

    print("[Analysis Agent] Generating response...")

    system_prompt = f"""
You are an AI Analysis Agent.

Use the following context while answering.

Context:
{state['retrieved_docs']}

Previous Feedback:
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
# Validation Schema
# -----------------------------
class ValidationOutput(BaseModel):

    is_valid: bool = Field(
        description="True if the response is correct."
    )

    feedback: str = Field(
        description="Suggestions if response is incorrect."
    )


# -----------------------------
# Validation Agent
# -----------------------------
def validation_agent(state: GraphState):

    print("[Validation Agent] Validating response...")

    validator = llm.with_structured_output(ValidationOutput)

    messages = [
        SystemMessage(
            content="Validate the response. Check correctness and relevance."
        ),
        HumanMessage(
            content=f"""
User Query:
{state['user_query']}

Generated Response:
{state['draft_response']}
"""
        )
    ]

    result = validator.invoke(messages)

    print("Validation:", result.is_valid)

    return {
        "is_valid": result.is_valid,
        "critic_feedback": result.feedback,
        "retry_count": state["retry_count"] + 1
    }


# -----------------------------
# Routing Function
# -----------------------------
def router(state: GraphState):

    if state["is_valid"]:
        print("Response Approved.")
        return "approved"

    if state["retry_count"] >= 3:
        print("Maximum retries reached.")
        return "approved"

    print("Retrying Analysis...")
    return "retry"


# -----------------------------
# Build Graph
# -----------------------------
workflow = StateGraph(GraphState)

workflow.add_node("ResearchAgent", research_agent)
workflow.add_node("AnalysisAgent", analysis_agent)
workflow.add_node("ValidationAgent", validation_agent)

workflow.set_entry_point("ResearchAgent")

workflow.add_edge("ResearchAgent", "AnalysisAgent")
workflow.add_edge("AnalysisAgent", "ValidationAgent")

workflow.add_conditional_edges(
    "ValidationAgent",
    router,
    {
        "approved": END,
        "retry": "AnalysisAgent"
    }
)

app = workflow.compile()


# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":

    initial_state = {
        "user_query": "Explain Agentic AI using LangGraph.",
        "retrieved_docs": "",
        "draft_response": "",
        "critic_feedback": "",
        "retry_count": 0,
        "is_valid": False
    }

    print("\n========== Agentic AI Workflow ==========")

    result = app.invoke(initial_state)

    print("\n========== Final Response ==========\n")
    print(result["draft_response"])
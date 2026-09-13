import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
import streamlit as st

load_dotenv()

st.set_page_config(page_title="LinkedIn Post Generator", page_icon=None)

#we are going to make tool

search_tool=TavilySearch(max_results=3)

tools=[search_tool]

# llm

#writer
writer_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.6)
writer_llm_with_tools=writer_llm.bind_tools(tools)

#revier
reviewer_llm=ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


#state building

class State(TypedDict):
    topic:str
    messages:Annotated[list,add_messages]
    draft:str
    review_feedback:str
    is_approved:bool
    attempt:int


#nodes

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150-200 words, ends with a question or call_to _action to invite"
    "engagement. Do not use hashtags."
)

# writer node

def writer_node(state: State) -> dict:
    """write (or rewrite) the LinkedIn post.Can call tavily to search first."""
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state["review_feedback"]

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}\n"
            "If you need current information, search the web first."
        )
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n"
            f"Here is the reviewer's feedback:\n\n{previous_feedback}\n\n"
            "Write a new, improved draft that fixes every issue mentioned.\n"
            "Do not repeat the same mistakes."
        )

    messages = [("system", WRITER_SYSTEM_PROMPT)]
    if state.get("messages"):
        messages += state["messages"]
    messages.append(("human", user_message))

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }

tool_node=ToolNode(tools)

def extract_draft_node(state:State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state['messages'][-1]
    content = last_message.content

    if isinstance(content, list):
        draft = " ".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    else:
        draft = content

    return {"draft": draft}


REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim - uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
    "8. Directly and clearly addresses the given topic\n"

)

def reviewer_node(state:State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['draft']

    prompt = (
        f"review this LinkedIn post draft : \n"
        f"{draft}\n"
        f"give your reviews"
    )
    response=reviewer_llm.invoke(
        [("system",REVIEWER_SYSTEM_PROMPT),("human",prompt)]
    )
    review_text = response.content.strip()

    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    return {
    "review_feedback": feedback,
    "is_approved": is_approved,
}

#router function

def should_use_tool(state: State):
    last_message = state['messages'][-1]

    if getattr(last_message, 'tool_calls', None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state: State):
    if state['is_approved']:
        return END
    if state['attempt'] >= 3:
        return END
    return "writer"


@st.cache_resource
def build_graph():
    #build the graph
    graph = StateGraph(State)

    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START,'writer')

    graph.add_conditional_edges(
        "writer",should_use_tool
    )
    graph.add_edge("tools","writer")
    graph.add_edge("extract_draft","reviewer")

    graph.add_conditional_edges(
        "reviewer",should_stop_looping
    )

    return graph.compile()

app = build_graph()

# ---------------- Streamlit UI ----------------

st.title("Welcome to the LinkedIn Post Generator")
st.write(
    "This tool will draft a LinkedIn post for you, review it "
    "itself, and iterate until it's publish-ready."
)

topic = st.text_input("What topic do you want a LinkedIn post about?")

if st.button("Generate Post", disabled=not topic.strip()):
    initial_state = {
        "topic": topic.strip(),
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    progress_box = st.status("Starting generation...", expanded=True)

    # accumulate state updates as the graph streams, so we run the graph ONLY ONCE
    final_state = dict(initial_state)

    for step in app.stream(initial_state, config={"recursion_limit": 25}):
        for node_name, node_output in step.items():
            final_state.update(node_output)

            if node_name == "writer":
                progress_box.write(f"Attempt {final_state.get('attempt', '?')}: writer drafting the post...")
            elif node_name == "tools":
                progress_box.write("Searching the web for fresh context...")
            elif node_name == "extract_draft":
                progress_box.write("Draft extracted, sending to reviewer...")
            elif node_name == "reviewer":
                verdict = "APPROVED" if node_output.get("is_approved") else "REJECTED"
                progress_box.write(f"Reviewer verdict: {verdict}")
                progress_box.write(f"Feedback: {node_output.get('review_feedback', '')}")

    if final_state["is_approved"]:
        progress_box.update(label="Post approved!", state="complete")
    else:
        progress_box.update(label="Reached max attempts without approval", state="error")

    st.subheader("Final LinkedIn Post")
    st.text_area("Post", final_state["draft"], height=300, label_visibility="collapsed")

    col1, col2 = st.columns(2)
    col1.metric("Total attempts", final_state["attempt"])
    col2.metric("Approved", "Yes" if final_state["is_approved"] else "No")
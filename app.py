import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="College Assistant", page_icon=None)
st.title("Welcome to the College assistant")

# system 1 building the RAG retriever

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

with st.spinner("Loading embedding model (first run can take a minute)..."):
    embeddings = get_embeddings()

def build_retriever(pdf_path:str):

    loader=PyPDFLoader(pdf_path)

    document=loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)

    chunks=splitter.split_documents(document)

    vector_store=FAISS.from_documents(chunks,embeddings)

    return vector_store.as_retriever(search_kwargs={'k':4})

@st.cache_resource
def get_retrievers():
    academic_retriever=build_retriever("academics_handbook.pdf")
    fee_retriever=build_retriever("fee_structure.pdf")
    return academic_retriever, fee_retriever

with st.spinner("Loading and indexing PDFs..."):
    try:
        academic_retriever, fee_retriever = get_retrievers()
    except Exception as e:
        st.error("Could not load the PDF files.")
        st.exception(e)
        st.stop()


llm=ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


#stage 2 state
class State(TypedDict):
    pragramme:str
    messages:Annotated[list,add_messages]
    query_type:str
    retrieved_context:str


#stage 3 creating nodes

def classifier_node(state:State) -> dict:
    """look at the latest message and dicide which path we take."""

    last_message=state['messages'][-1].content

    prompt = (
    "Classify the following student query into exactly one category: "
    "'academic', 'fee', or 'general'.\n\n"
    "Use 'academic' for questions about attendance, exams, grading, credits, "
    "promotion, course structure, summer training, or degree requirements.\n"
    "Use 'fee' for questions about tuition, payment, refund, late charges, "
    "scholarships, or any money-related topic.\n"
    "Use 'general' for greetings, casual talk, or anything not related to "
    "the college rules or fee.\n\n"
    f"Query: {last_message}\n\n"
    "Return only one word: academic, fee, or general."
)
    response=llm.invoke(prompt)
    category=response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return{"query_type":category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible. "
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


#step 4 - router function

def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


#stage 5  building the graph

@st.cache_resource
def build_graph():
    graph=StateGraph(State)

    graph.add_node("classifier",classifier_node)
    graph.add_node("academic_rag",academic_rag_node)
    graph.add_node("fee_rag",fee_rag_node)
    graph.add_node("general",general_node)
    graph.add_node("response",response_node)

    # stage 6 connecting the edges
    graph.add_edge(START,"classifier")

    graph.add_conditional_edges(
        "classifier",route_query
    )

    graph.add_edge("academic_rag","response")
    graph.add_edge("fee_rag","response")
    graph.add_edge("general","response")

    graph.add_edge("response",END)

    return graph.compile()

with st.spinner("Setting up the assistant..."):
    app = build_graph()

# ---------------- Streamlit UI ----------------

if "student_programme" not in st.session_state:
    st.session_state.student_programme = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.student_programme is None:
    st.write("Which programme are you in?")
    choice = st.radio(
        "Select your programme",
        options=["1", "2", "3"],
        format_func=lambda c: {"1": "BCA", "2": "BBA", "3": "B.com (H)"}[c],
    )
    if st.button("Enter 1, 2 or 3"):
        programme_map = {
            "1": "BCA",
            "2": "BBA",
            "3": "B.Com (H)"
        }
        student_programme = programme_map.get(choice, "BCA")
        st.session_state.student_programme = student_programme
        st.success(f"Great! You're set as a {student_programme} student.")
        st.rerun()
else:
    st.caption(f"Programme: {st.session_state.student_programme}")

    for role, content in st.session_state.messages:
        with st.chat_message("user" if role == "human" else "assistant"):
            st.write(content)

    user_query = st.chat_input("You:")

    if user_query:
        if user_query.lower() in ["exit", "quit"]:
            st.session_state.student_programme = None
            st.session_state.messages = []
            st.rerun()
        else:
            st.session_state.messages.append(("human", user_query))
            with st.chat_message("user"):
                st.write(user_query)

            result = app.invoke({
                "programme": st.session_state.student_programme,
                "messages": [("human", user_query)]
            })

            assistant_reply = result['messages'][-1].content
            st.session_state.messages.append(("ai", assistant_reply))
            with st.chat_message("assistant"):
                st.write(f"Assistant:{assistant_reply}")
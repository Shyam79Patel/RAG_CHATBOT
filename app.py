import streamlit as st
import os
import time
from rag_pipeline import build_vectorstore, load_vectorstore, get_qa_chain, get_answer

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🧠",
    layout="centered")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background-color: #0e1117;
    color: white;}

/* Chat bubbles */
.user {
    background: #1f6feb;
    padding: 10px 14px;
    border-radius: 10px;
    margin: 6px 0;
    width: fit-content;
    margin-left: auto;}

.bot {
    background: #30363d;
    padding: 10px 14px;
    border-radius: 10px;
    margin: 6px 0;
    width: fit-content;}

/* Input */
.stTextInput input {
    background: #161b22 !important;
    color: white !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_pipeline():
    status = st.empty()

    CHROMA_DIR = "chroma_db"

    if not os.path.exists(CHROMA_DIR):
        status.info("🔄 Building vector store...")
        vs = build_vectorstore()
    else:
        status.info("📦 Loading vector store...")
        vs = load_vectorstore()

    status.info("⚙️ Initializing QA chain...")
    chain, retriever = get_qa_chain(vs)

    status.success("✅ System ready")

    time.sleep(0.5)
    status.empty()

    return chain, retriever

chain, retriever = init_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 Research Assistant")
st.caption("RAG · LLM · ChromaDB")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot">{msg["content"]}</div>', unsafe_allow_html=True)

query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user">{query}</div>', unsafe_allow_html=True)

    response_placeholder = st.empty()

    with st.status("Processing...", expanded=True) as status:
        st.write("🔍 Retrieving documents...")
        time.sleep(0.5)

        st.write("🧠 Generating answer...")
        result = get_answer(query, chain, retriever)

        status.update(label="✅ Done", state="complete")

    answer = result["answer"]

    streamed_text = ""
    for word in answer.split():
        streamed_text += word + " "
        response_placeholder.markdown(f'<div class="bot">{streamed_text}</div>', unsafe_allow_html=True)
        time.sleep(0.02)

    # save final response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer})

import os
import warnings
import logging
import streamlit as st

# --- Modern LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# CONFIG
# -----------------------------

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="Parallel & Distributed Computing Assistant",
    page_icon="💻",
    layout="centered"
)

# -----------------------------
# MODERN MINIMAL UI
# -----------------------------

st.markdown(
    """
<style>

:root{
    --iu-blue:#003366;
    --iu-gold:#f8a51b;
}

#MainMenu,
footer,
header{
    visibility:hidden;
}

.stApp{
    background:#f8fafc;
}

/* Hero */

.hero-title{
    text-align:center;
    font-size:2.2rem;
    font-weight:700;
    color:var(--iu-blue);
    margin-top:10px;
    margin-bottom:5px;
}

.hero-subtitle{
    text-align:center;
    color:#64748b;
    font-size:1rem;
    margin-bottom:30px;
}

/* Chat Messages */

[data-testid="stChatMessage"]{
    border-radius:18px;
    padding:18px;
    border:none;
    box-shadow:0 2px 12px rgba(0,0,0,0.05);
    margin-bottom:15px;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
    background:#eef6ff;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
    background:#fffdf7;
}

[data-testid="stChatMessageAvatarUser"]{
    background:#003366 !important;
}

[data-testid="stChatMessageAvatarAssistant"]{
    background:#f8a51b !important;
}

/* Chat Input */

textarea{
    border-radius:15px !important;
    border:2px solid #dbeafe !important;
}

textarea:focus{
    border:2px solid #003366 !important;
    box-shadow:none !important;
}

/* Buttons */

.stButton button{
    border-radius:12px;
    border:1px solid #dbeafe;
    background:white;
    color:#003366;
    font-weight:500;
}

.stButton button:hover{
    border-color:#f8a51b;
}

/* Text */

h1,h2,h3,h4,p,span,div{
    color:#003366;
}

</style>

<div class="hero-title">
Parallel & Distributed Computing Assistant
</div>

<div class="hero-subtitle">
Ask questions from your PDC course material
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# CHAT HISTORY
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -----------------------------
# VECTOR STORE
# -----------------------------

@st.cache_resource
def get_vectorstore():

    pdf_file = "pdc.pdf"

    if not os.path.exists(pdf_file):
        return None

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore

# -----------------------------
# INPUT
# -----------------------------

prompt = st.chat_input(
    "Ask a question about Parallel & Distributed Computing..."
)

# -----------------------------
# SUGGESTED QUESTIONS
# -----------------------------

if len(st.session_state.messages) == 0 and not prompt:

    st.markdown(
        "<center><b>Suggested Questions</b></center>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "What is Parallel Computing?",
            use_container_width=True
        ):
            prompt = "What is Parallel Computing?"

        if st.button(
            "Explain Flynn's Taxonomy",
            use_container_width=True
        ):
            prompt = "Explain Flynn's Taxonomy"

    with c2:

        if st.button(
            "Difference between Parallel and Distributed Computing",
            use_container_width=True
        ):
            prompt = (
                "Difference between Parallel and Distributed Computing"
            )

        if st.button(
            "Explain Amdahl's Law",
            use_container_width=True
        ):
            prompt = "Explain Amdahl's Law"

# -----------------------------
# CHAT PROCESSING
# -----------------------------

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    try:

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant"
        )

        vectorstore = get_vectorstore()

        if vectorstore is None:

            response_text = """
### Course Material Not Found

Make sure:

- `pdc.pdf` exists
- `pdc.pdf` is in the same folder as `app.py`
"""
            with st.chat_message("assistant"):
                st.markdown(response_text)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text
                }
            )

        else:

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            system_prompt = """
You are a helpful Parallel and Distributed Computing tutor.

Answer questions using only the provided course material.

Guidelines:
- Explain concepts clearly.
- Use examples when relevant.
- Keep answers concise and accurate.
- If a topic is not available in the document,
  say you could not find it in the course material.

Context:
{context}

Question:
{input}
"""

            prompt_template = ChatPromptTemplate.from_template(system_prompt)

            # Modern formatting utility to pack retrieved documents into clean string strings
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Retrieve source context blocks manually beforehand for evaluation
            source_docs = retriever.invoke(prompt)
            context_string = format_docs(source_docs)

            # Build declarative chain using standard LCEL composition
            rag_chain = prompt_template | llm | StrOutputParser()

            # Process prediction execution
            response_text = rag_chain.invoke(
                {
                    "context": context_string,
                    "input": prompt
                }
            )

            # -------------------------
            # MATCH SCORE
            # -------------------------

            score = 90

            if source_docs:

                query_words = set(
                    prompt.lower().split()
                )

                if query_words:

                    best_match_count = 0

                    for doc in source_docs:

                        matches = sum(
                            1
                            for word in query_words
                            if word in doc.page_content.lower()
                        )

                        best_match_count = max(
                            best_match_count,
                            matches
                        )

                    conf = int(
                        (
                            best_match_count
                            / len(query_words)
                        )
                        * 100
                    )

                    score = max(
                        45,
                        min(
                            98,
                            conf + 35
                        )
                    )

            # Render HTML elements without saving them in history storage structures
            with st.chat_message("assistant"):
                st.markdown(response_text)
                
                bar_html = f"""
                <div style="margin-top:18px; font-family: sans-serif;">
                    <div style="font-size:13px; color:#64748b; margin-bottom:6px;">
                        Response Match
                    </div>
                    <div style="background:#e5e7eb; height:8px; border-radius:20px; overflow:hidden;">
                        <div style="width:{score}%; height:100%; background:#f8a51b;"></div>
                    </div>
                    <div style="text-align:right; font-size:12px; margin-top:4px; color:#003366; font-weight:600;">
                        {score}%
                    </div>
                </div>
                """
                st.html(bar_html)

            # Record conversation safely
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text
                }
            )

    except Exception as e:

        st.error(
            f"System Error: {str(e)}"
        )

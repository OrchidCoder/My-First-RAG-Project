
import os
import tempfile
from pathlib import Path

import streamlit as st

from main import (
    PDF_PATH,
    create_rag_application,
    query_rag,
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="My First RAG Project",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 2. CUSTOM WEBSITE DESIGN
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        color: #2563eb;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 15px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 3. API KEY CONFIGURATION
# ==========================================================

def get_api_key():

    # First try Streamlit Cloud secrets
    try:
        key = st.secrets.get("NVIDIA_API_KEY", "")

        if key:
            return key

    except Exception:
        pass

    # Local development fallback
    return os.getenv("NVIDIA_API_KEY", "")


# ==========================================================
# 4. CREATE AND CACHE RAG RESOURCES
# ==========================================================

@st.cache_resource(show_spinner=False)
def build_rag_resources(pdf_bytes, api_key):

    # Create a temporary PDF for processing
    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = Path(temp_dir) / "uploaded_document.pdf"

        pdf_path.write_bytes(pdf_bytes)

        # Build the existing RAG application
        vectorstore, rag_chain = create_rag_application(
            pdf_path=pdf_path,
            api_key=api_key,
        )

    return vectorstore, rag_chain


# ==========================================================
# 5. WEBSITE HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">📚 My First RAG Project</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Intelligent PDF-Based Question Answering System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
    <h3>🤖 Welcome to your AI PDF Assistant!</h3>
    Ask questions about your PDF and receive accurate,
    document-based answers powered by NVIDIA Nemotron.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 6. SIDEBAR - DOCUMENT SELECTION
# ==========================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 📄 Select your document")

default_pdf_exists = Path(PDF_PATH).is_file()

if default_pdf_exists:

    pdf_mode = st.sidebar.radio(
        "Choose PDF source:",
        [
            "Use default STATS.pdf",
            "Upload my own PDF",
        ],
    )

else:

    st.sidebar.warning(
        "Default STATS.pdf is not available on this server."
    )

    pdf_mode = "Upload my own PDF"

    st.sidebar.info(
        "Upload a PDF to start asking questions."
    )


uploaded_file = None

if pdf_mode == "Upload my own PDF":

    uploaded_file = st.sidebar.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Choose a PDF document to ask questions about.",
    )

    if uploaded_file:

        st.sidebar.success(
            f"Selected: {uploaded_file.name}"
        )


# ==========================================================
# 7. GET SELECTED PDF
# ==========================================================

pdf_bytes = None
document_name = None

if pdf_mode == "Use default STATS.pdf":

    if default_pdf_exists:

        pdf_bytes = Path(PDF_PATH).read_bytes()

        document_name = "STATS.pdf"

elif uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    document_name = uploaded_file.name


# ==========================================================
# 8. API KEY CHECK
# ==========================================================

api_key = get_api_key()

if not api_key:

    st.error(
        "NVIDIA API key is missing. "
        "Configure NVIDIA_API_KEY in your .env file "
        "or Streamlit Secrets."
    )

    st.stop()


# ==========================================================
# 9. DOCUMENT STATUS
# ==========================================================

if pdf_bytes is None:

    st.info(
        "👈 Select your PDF from the sidebar to begin."
    )

    st.markdown(
        """
        ### How to use this application

        1. Select the default PDF or upload your own.
        2. Wait for document indexing to finish.
        3. Enter a question in the chat box.
        4. Read the detailed answer.
        5. Review the simple final summary.
        """
    )

    st.stop()


# ==========================================================
# 10. PREPARE DOCUMENT INDEX
# ==========================================================

# A new document selection starts a new chat session.
document_signature = (
    document_name,
    len(pdf_bytes),
    hash(pdf_bytes),
)

if st.session_state.get("document_signature") != document_signature:

    st.session_state.document_signature = document_signature

    st.session_state.messages = []

    st.session_state.rag_resources = None


if st.session_state.get("rag_resources") is None:

    with st.spinner(
        "📖 Reading PDF, creating embeddings and indexing document..."
    ):

        try:

            vectorstore, rag_chain = build_rag_resources(
                pdf_bytes,
                api_key,
            )

            st.session_state.rag_resources = (
                vectorstore,
                rag_chain,
            )

            st.success(
                f"Successfully processed {document_name}!"
            )

        except Exception as error:

            st.error(
                f"Unable to process PDF: {error}"
            )

            st.stop()


# ==========================================================
# 11. DOCUMENT INFORMATION
# ==========================================================

st.markdown("### 📄 Current Document")

st.success(
    f"Active document: {document_name}"
)

st.caption(
    "Your questions will be checked against this document "
    "before an answer is generated."
)


# ==========================================================
# 12. CHAT HISTORY
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


st.markdown("### 💬 Ask Your Questions")


# Display existing conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================================
# 13. LIVE QUESTION AND ANSWER
# ==========================================================

question = st.chat_input(
    "Ask anything related to your PDF..."
)


if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Checking document and generating answer..."
        ):

            try:

                vectorstore, rag_chain = (
                    st.session_state.rag_resources
                )

                answer = query_rag(
                    question,
                    vectorstore,
                    rag_chain,
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Unable to generate answer: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# ==========================================================
# 14. CLEAR CHAT
# ==========================================================

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ==========================================================
# 15. FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    **My First RAG Project**

    Powered by LangChain • Qdrant • FastEmbed • NVIDIA Nemotron

    PDF-based Question Answering System

    </div>
    """,
    unsafe_allow_html=True,
)
